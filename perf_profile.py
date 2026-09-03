# -*- coding: utf-8 -*-
"""
perf_profile.py — GPU 硬件分级检测与自适应性能配置 (P4)。

启动时用 nvidia-smi 读取显卡型号/显存，分 低/中/高/旗舰 四档；每档对应一份
性能 profile(融合分辨率/纹理核/时域复用步长/EMA)，供 skin_mask 应用。
运行时 Governor 按 GPU 利用率+帧耗时做滞后升降档(不越过启动检测的原始档)。
无 nvidia-smi/无 NVIDIA GPU -> tier "none"，上层应切纯 CPU 回退管线。
"""
import os
import re
import shutil
import subprocess

# 各档参数(720p 实测定档, 见 bench_tier2.py):
#   fusion_side  肤色融合通道边长。全档固定 768: 隔离实验显示 768 与 1024 的输出
#                遮罩 IoU 0.987/0.985(几乎同一张图), 但 1024 每帧多花 18~65ms —— 
#                加大它买不到质量, 只买到耗时, 故不作为档位阶梯。
#   tex_k        纹理门控核尺寸(低配 3, 其余 5)
#   seg_stride   分割每 N 帧跑一次。相对每帧分割的 IoU: N=2 -> 0.920, N=3 -> 0.864,
#                是次级质量杠杆(代价也次级)。
#   raw_stride   整遮罩每 N 帧算一次, 中间帧复用。质量的主杠杆: 运动 8px/帧的序列上
#                N=2 相对 N=1 的 IoU 仅 0.351(运镜时皮肤区明显滞后)。
#   ema_alpha    时域 EMA 上一帧权重(stride 越大越高, 补偿复用引入的滞后)
# 档位阶梯按 50 系为主流上调: 5070~5090 至少落在 high(每帧全量重算遮罩 + 分割隔帧),
# 5080/5090/4090 类落 ultra(分割也每帧跑, 质量上限)。
# 注意: 遮罩跑在 CPU(mediapipe), 720p 每帧全量约 105~160ms, 而 GPU 双通道只要
# 33ms/帧 —— 高档位的瓶颈在 CPU 遮罩, 不在显卡。
PROFILES = {
    "low":   dict(fusion_side=768, tex_k=3, seg_stride=3, raw_stride=3, ema_alpha=0.7),
    "mid":   dict(fusion_side=768, tex_k=5, seg_stride=3, raw_stride=2, ema_alpha=0.6),
    "high":  dict(fusion_side=768, tex_k=5, seg_stride=2, raw_stride=1, ema_alpha=0.6),
    "ultra": dict(fusion_side=768, tex_k=5, seg_stride=1, raw_stride=1, ema_alpha=0.5),
    "none":  dict(fusion_side=768, tex_k=3, seg_stride=4, raw_stride=3, ema_alpha=0.7),
}
ORDER = ("low", "mid", "high", "ultra")
TIER_NAME = {"low": "低配", "mid": "中配", "high": "高配", "ultra": "旗舰",
             "none": "无GPU(CPU回退)"}

# ---- P6: 遮罩多实例交错并行 ----
# 遮罩跑在 CPU, 720p 单帧约 110ms, 而 GPU 双通道只要 33ms —— 瓶颈在 CPU。多个
# SkinMasker 实例跑交错帧可近线性提速(720p 实测 K2 1.84x / K3 2.41~2.56x / K4 3.07x)。
# EMA 不能留在实例里: 每实例一条 EMA 链 -> 运动补偿后的闪烁升到串行的 1.41~1.67 倍
# (按 i%K 分组可见周期 K 的规律起伏); 改在有序消费点用单一共享 prev 递推, 输出与
# 串行逐位一致(实测最大差 0, 闪烁 1.00x)。每多一个实例约 +130MB 内存, 且 mediapipe
# 的 close() 要干等 ~100s 才返回(期间同进程 mask() 掉到 2.2 倍) —— 实例只能缓存复用
# 不能随手关, 所以路数必须按 档位/CPU 核数/帧面积 卡上限。
MASK_WORKERS = {"low": 1, "mid": 2, "high": 3, "ultra": 3, "none": 1}


def mask_workers(tier, pixels=0, cpu=None):
    """该档位建议的遮罩实例数(>=1)。
    - 每实例满载吃 1~2 个核, 且 ffmpeg 编码也要吃核, 故不超过 cpu//4
    - 帧面积 >= 6M 像素(4K 级)时压到 2: 环形缓冲与模型内存都随 K 线性涨
    - 实例一旦建了就一直常驻(不能关, 见上方注释), 所以宁愿保守
    """
    k = MASK_WORKERS.get(tier, 1)
    k = max(1, min(k, (cpu or os.cpu_count() or 1) // 4))
    if pixels >= 6000000:
        k = min(k, 2)
    return k


def parallel_profile(tier, k):
    """返回 (profile, ema_alpha): 给遮罩实例用的 profile 与应在有序点做的 EMA 权重。

    profile 里的 ema_alpha 总是 0 —— EMA 改由调用方在严格帧序的消费点统一做。
    该变换对 K=1 也适用: EMA 是 mask() 返回前的最后一步, 搬到外部递推结果不变。
    K>1时把时域抽帧关掉(seg/raw stride=1): 质量阶梯改由并行买单 —— raw_stride=2
    相对 1 的 IoU 只有 0.351, 是质量最大的破坏源, 能不用就不用。
    """
    prof = dict(PROFILES.get(tier, PROFILES["mid"]))
    alpha = float(prof.get("ema_alpha", 0.6))
    if k > 1:
        prof["seg_stride"] = 1
        prof["raw_stride"] = 1
    prof["ema_alpha"] = 0.0
    return prof, alpha


_HIDE_WINDOW = 0x08000000   # CREATE_NO_WINDOW


def detect_gpu(log=None):
    """返回 (tier, desc)。失败/无 NVIDIA GPU 时 tier="none"。"""
    log = log or (lambda m: None)
    smi = shutil.which("nvidia-smi")
    if not smi:
        return "none", "未检测到 nvidia-smi(无 NVIDIA GPU 或未装驱动)"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, timeout=10, creationflags=_HIDE_WINDOW
        ).stdout.decode("utf-8", "replace").strip()
        name, mem = [x.strip() for x in out.splitlines()[0].split(",")]
        mem = int(mem)
    except Exception as ex:
        return "none", "nvidia-smi 查询失败: %s" % str(ex)[:80]
    n = name.upper()
    m = re.search(r"(RTX|GTX)\s?(\d{4})", n)
    fam, num = (m.group(1), int(m.group(2))) if m else ("", 0)
    gen, model = num // 1000, num % 1000     # 5070 -> gen 5, model 70
    if fam == "RTX" and mem >= 15000 and gen >= 4 and model >= 80:
        tier = "ultra"                       # 4080/4090/5080/5090 类
    elif fam == "RTX" and mem >= 10000 and (
            (gen >= 4 and model >= 70) or (gen == 3 and model >= 80)):
        tier = "high"                        # 5070/5070Ti/4070/3080 类
    elif fam == "RTX" or num >= 1660 or mem >= 6000:
        tier = "mid"
    else:
        tier = "low"
    return tier, "%s / 显存 %d MB" % (name, mem)


def gpu_util():
    """当前 GPU 利用率(%); 失败返回 None。每 2s 调一次开销可忽略。"""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, timeout=8, creationflags=_HIDE_WINDOW
        ).stdout.decode("utf-8", "replace").strip().splitlines()
        return int(out[0])
    except Exception:
        return None


# ---- 按 GPU 系列挑 nvngx_dlssnr.dll ----
# 30系/40系/50系目录各放一份适配该代的渲染库(型号名里的 RTX 30xx/40xx/50xx)。
# 候选位置: 工具目录\<系列>\ 与 工具目录的上一级\<系列>\(当前分发布局是后者)。
# NGX 每进程只能初始化一次, 选择必须在首次 init 前生效(见 dlss_engine.set_plugin_dll)。
NVNGX_SERIES_DIR = {3: "30系", 4: "40系", 5: "50系"}


def nvngx_dll_for(gpu_desc, tool_dir, exe_dir=None):
    """按 GPU 型号描述(如 "NVIDIA GeForce RTX 4080 / 显存 16376 MB")挑插件 DLL。
    返回 (路径, 系列名或None)。型号不在 30~50 系或对应目录里没放 dll 时回退
    tool_dir 自带的 nvngx_dlssnr.dll(系列名=None 表示回退)。
    exe_dir: 冻结(PyInstaller)时 tool_dir=_MEIPASS 是临时目录, 系列附加包实际
    放在 exe 旁边, 由调用方把 exe 目录传进来一并搜索。"""
    fallback = os.path.join(tool_dir, "nvngx_dlssnr.dll")
    m = re.search(r"RTX\s?(\d{4})", (gpu_desc or "").upper())
    sdir = NVNGX_SERIES_DIR.get(int(m.group(1)) // 1000) if m else None
    if sdir:
        bases = [tool_dir, os.path.dirname(tool_dir)]
        if exe_dir:
            bases.append(exe_dir)
        for base in bases:
            p = os.path.join(base, sdir, "nvngx_dlssnr.dll")
            if os.path.exists(p):
                return p, sdir
    return fallback, None


class Governor:
    """运行时升降档治理器(带滞后, 防抖动)。
    每次拿到 (利用率, 帧耗时达标与否) 就 sample() 一次; 发生档位变化回调
    on_change(new_tier, reason)。升档最多回到启动检测的原始档, 降档可到 low。"""

    def __init__(self, base_tier, on_change=None):
        self.base = base_tier if base_tier in ORDER else "mid"
        self.tier = self.base
        self._on_change = on_change or (lambda t, r: None)
        self._low_cnt = 0
        self._high_cnt = 0

    def sample(self, util, fps_ok):
        if util is None or self.tier == "low":
            return self.tier
        if util < 30 and not fps_ok:
            self._low_cnt += 1
            self._high_cnt = 0
            if self._low_cnt >= 3:           # 连续 3 次不达标 -> 降一档
                self._low_cnt = 0
                self.tier = ORDER[max(0, ORDER.index(self.tier) - 1)]
                self._on_change(self.tier, "GPU利用率<30%% 且帧率不达标, 降级到 %s" % TIER_NAME[self.tier])
        elif util > 85 and fps_ok:
            self._high_cnt += 1
            self._low_cnt = 0
            if self._high_cnt >= 5 and self.tier != self.base:  # 连续 5 次富余 -> 升回一档
                self._high_cnt = 0
                self.tier = ORDER[min(len(ORDER) - 1, ORDER.index(self.tier) + 1)]
                self._on_change(self.tier, "GPU满载且帧率富余, 提升到 %s" % TIER_NAME[self.tier])
        else:
            self._low_cnt = self._high_cnt = 0
        return self.tier
