#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skin_mask.py — 从视频/图片帧中提取"人物皮肤"遮罩 (float32, 0..1)。

用于 DLSS5 工具的"皮肤双通道"渲染：把 skin_struct 增强的效果限制在皮肤区域。

后端优先级：
  1. MediaPipe (models/ 下的 selfie_multiclass_256x256 + blaze_face_short_range
     + blaze_face_full_range[可选，全身照小脸检出])
     - multiclass 分割类别: 0 背景 / 1 头发 / 2 身体皮肤 / 3 脸部皮肤 / 4 衣服 / 5 其他
     - 人脸检测在 1024 高分辨率通道跑(short+full 双检测器)，画成椭圆补全脸颈皮肤
     - 分割模型对 CG/动漫角色常把躯干皮肤误判成衣服，故再融合
       “肤色判据 + 纹理门控”补全躯干/四肢皮肤(乘 person/非头发 防背景误报)
  2. OpenCV 回退 (mediapipe 或模型缺失时): Haar 人脸检测 + 肤色判据 + 纹理门控

遮罩输出前统一做：小图推理→放大→阈值化→形态学去噪→羽化(高斯模糊)→视频时域 EMA 平滑。
"""
import os
import sys

import cv2
import numpy as np

try:
    cv2.setNumThreads(4)   # 限制 OpenCV 并行, 降低与 tflite XNNPACK 线程竞争导致 AV 的风险
except Exception:
    pass

if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_BASE, "models")
SEG_MODEL = os.path.join(MODELS_DIR, "selfie_multiclass_256x256.tflite")
FACE_MODEL = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
FACE_MODEL_FULL = os.path.join(MODELS_DIR, "blaze_face_full_range.tflite")

# selfie_multiclass_256x256 类别号（官方定义）
SKIN_CLASSES = (2, 3)   # 2=body-skin, 3=face-skin

# 肤色判据可调参数：默认值 = 调参实验定下的硬编码值；
# UI 滑条范围见 SKIN_PARAM_LIMITS（超范围由调用方钳制）
DEFAULT_SKIN_PARAMS = {
    'cr_c': 148.0, 'cr_w': 24.0,   # Cr 中心 / 容差
    'cb_c': 117.0, 'cb_w': 16.0,   # Cb 中心 / 容差
    'h_c': 8.0, 'h_w': 14.0,       # 色相中心 / 容差 (0-180 环绕)
    's_lo': 15.0, 's_hi': 220.0,   # 饱和度下界 / 上界
    'v_lo': 110.0,                 # 亮度下界
}
SKIN_PARAM_LIMITS = {
    'cr_c': (0.0, 255.0), 'cr_w': (4.0, 64.0),
    'cb_c': (0.0, 255.0), 'cb_w': (4.0, 64.0),
    'h_c': (0.0, 180.0), 'h_w': (2.0, 45.0),
    's_lo': (0.0, 200.0), 's_hi': (100.0, 255.0),
    'v_lo': (0.0, 255.0),
}


def norm_skin_params(params=None):
    """合并用户参数与默认值，并按 SKIN_PARAM_LIMITS 钳制；返回完整 dict。"""
    p = dict(DEFAULT_SKIN_PARAMS)
    if params:
        for k in p:
            if k in params:
                try:
                    p[k] = float(params[k])
                except (TypeError, ValueError):
                    pass
    for k, (lo, hi) in SKIN_PARAM_LIMITS.items():
        p[k] = max(lo, min(hi, p[k]))
    return p

# 推理用最大边长（分割模型内部就是 256，再大也没意义）
_MAX_SIDE = 512
# 人脸检测通道边长：全身照脸很小，512 下检不出，用 1024 提高小脸召回
_DET_SIDE = 1024
# 融合通道边长(实例可被 perf profile 覆盖。实测 768 与 1024 的输出遮罩 IoU 0.987,
# 加大只多花 18~65ms/帧 而不提质量, 故各档统一 768)
_FUSION_SIDE = 768


def _channels(bgr):
    """一次算好 YCrCb/HSV float 通道，供 _color_skin_chs 复用(避免迭代重算)。"""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return ycrcb, hsv


def _color_skin_chs(chs, p=None):
    """肤色评分 0..1(预计算通道版)。判据见 DEFAULT_SKIN_PARAMS。"""
    p = norm_skin_params(p)
    ycrcb, hsv = chs
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    s_cr = np.clip(1.0 - np.abs(cr - p['cr_c']) / max(p['cr_w'], 1e-3), 0, 1)
    s_cb = np.clip(1.0 - np.abs(cb - p['cb_c']) / max(p['cb_w'], 1e-3), 0, 1)
    d_h = np.abs(hue - p['h_c'])
    s_h = np.clip(1.0 - np.minimum(d_h, 180.0 - d_h) / max(p['h_w'], 1e-3), 0, 1)
    s_s = np.clip((sat - p['s_lo']) / 25.0, 0, 1) * np.clip((p['s_hi'] - sat) / 60.0, 0, 1)
    s_v = np.clip((val - p['v_lo']) / 60.0, 0, 1)
    return np.clip(s_cr * s_cb * s_h * s_s * s_v * 1.8, 0, 1)


def _color_skin(bgr, p=None):
    """肤色评分 0..1。YCrCb + HSV 多因子软判据，参数 p 见 DEFAULT_SKIN_PARAMS，
    默认值兼顾真实皮肤(Cr~155)与 CG 浅肤(Cr~136-152)。
    注意 cv2.COLOR_BGR2YCrCb 的通道序是 Y / Cr / Cb。"""
    return _color_skin_chs(_channels(bgr), p)


def _tex_gate(bgr, k=5):
    """纹理门控 0..1：局部灰度标准差，平滑区(皮肤)通过，
    格纹/条纹/花纹(衣物)抑制。k 为核尺寸(低配 3 / 默认 5)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (k, k))
    sq = cv2.blur(gray * gray, (k, k))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return np.clip(1.0 - (std - 6) / 10.0, 0, 1)


class SkinMasker:
    """逐帧提取皮肤遮罩。视频连续帧传入时自动做时域平滑；
    拖动时间轴/换素材/暂停恢复时调用 reset() 清掉时域状态。"""

    def __init__(self, log=None):
        self._log = log or (lambda m: None)
        self.backend = None
        self._seg = None
        self._dets = []
        self._haar = None
        self._ema = None
        self._params = None
        # 性能 profile 可调项(见 perf_profile.PROFILES)
        self.fusion_side = _FUSION_SIDE
        self.tex_k = 5
        self.seg_stride = 3      # 分割每 N 帧跑一次, 中间帧复用 person/hair/模型皮肤
        self.raw_stride = 1      # 整遮罩每 N 帧算一次, 中间帧复用 raw(EMA 平滑)
        self.ema_alpha = 0.6     # 时域 EMA 上一帧权重
        self._frame_idx = 0
        self._seg_cache = None
        self._raw_cache = None
        self._init_backend()

    # ---------- backend ----------
    def _init_backend(self):
        try:
            if os.path.exists(SEG_MODEL) and os.path.exists(FACE_MODEL):
                from mediapipe.tasks.python import BaseOptions, vision
                self._seg = vision.ImageSegmenter.create_from_options(
                    vision.ImageSegmenterOptions(
                        base_options=BaseOptions(model_asset_path=SEG_MODEL),
                        output_confidence_masks=True))
                self._dets = []
                for model in (FACE_MODEL, FACE_MODEL_FULL):   # short 近脸强, full 补远距离小脸
                    if os.path.exists(model):
                        self._dets.append(vision.FaceDetector.create_from_options(
                            vision.FaceDetectorOptions(
                                base_options=BaseOptions(model_asset_path=model),
                                min_detection_confidence=0.35)))
                self.backend = "mediapipe"
                self._log("[皮肤遮罩] 后端: MediaPipe (multiclass 分割 + %d 个人脸检测器)" % len(self._dets))
                return
            self._log("[皮肤遮罩] 未找到模型文件 (%s)，回退 OpenCV" % MODELS_DIR)
        except Exception as ex:
            self._log("[皮肤遮罩] MediaPipe 不可用 (%s)，回退 OpenCV" % str(ex)[:120])
            self._seg = None
            self._dets = []
        try:
            haar_path = cv2.data.haarcascades + "frontalface_default.xml"
            # opencv-python 5.0.0.93 的 wheel 不带 Haar XML (issue #1244)，存在才用
            if os.path.exists(haar_path) and cv2.CascadeClassifier(haar_path).load(haar_path):
                self._haar = cv2.CascadeClassifier(haar_path)
            else:
                self._haar = None
                self._log("[皮肤遮罩] OpenCV 回退: 无 Haar 数据，仅用肤色判据")
            self.backend = "opencv"
        except Exception as ex:
            self._log("[皮肤遮罩] OpenCV Haar 也不可用: " + str(ex)[:120])
            self.backend = None

    def reset(self):
        """清空时域平滑状态（跳帧/换素材/暂停后调用）。"""
        self._ema = None
        self._seg_cache = None
        self._raw_cache = None
        self._frame_idx = 0

    def apply_profile(self, prof):
        """应用性能 profile(dict: fusion_side/tex_k/seg_stride/raw_stride/ema_alpha)，
        缺失键保持现值；切换后清缓存。"""
        if not prof:
            return
        self.fusion_side = int(prof.get('fusion_side', self.fusion_side))
        self.tex_k = int(prof.get('tex_k', self.tex_k)) | 1
        self.seg_stride = max(1, int(prof.get('seg_stride', self.seg_stride)))
        self.raw_stride = max(1, int(prof.get('raw_stride', self.raw_stride)))
        self.ema_alpha = float(prof.get('ema_alpha', self.ema_alpha))
        self._seg_cache = None
        self._raw_cache = None

    def close(self):
        for obj in [self._seg] + list(self._dets):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._seg = None
        self._dets = []

    # ---------- public ----------
    def mask(self, bgr, params=None):
        """输入 BGR 帧，返回同尺寸 float32 皮肤遮罩 (0..1)。不可用时返回全零。
        params 可覆盖肤色判据参数(见 DEFAULT_SKIN_PARAMS)，参数变化时重置时域平滑。"""
        if bgr is None:
            return np.zeros((1, 1), np.float32)
        if self.backend is None:
            return np.zeros(bgr.shape[:2], np.float32)
        p = norm_skin_params(params)
        if self._params is not None and any(abs(p[k] - self._params[k]) > 1e-6 for k in p):
            self._ema = None
            self._raw_cache = None     # 融合依赖参数, 判据变了不能复用 raw
        self._params = p
        h, w = bgr.shape[:2]
        idx = self._frame_idx
        self._frame_idx += 1
        if self._raw_cache is not None and self.raw_stride > 1 and (idx % self.raw_stride):
            raw = self._raw_cache
        else:
            raw = self._raw_mask(bgr, p)
            self._raw_cache = raw
        m = self._postprocess(raw, w, h)
        # 时域 EMA，抑制视频中的边缘抖动
        if self._ema is not None and self._ema.shape == m.shape:
            m = self.ema_alpha * self._ema + (1.0 - self.ema_alpha) * m
        self._ema = m
        return m

    def overlay(self, bgr, mask, alpha=0.45):
        """调试用：在 BGR 帧上叠加绿色遮罩可视化（返回新数组）。"""
        out = bgr.copy()
        m3 = mask[..., None]
        green = np.zeros_like(out)
        green[..., 1] = 255
        out = out.astype(np.float32) * (1 - m3 * alpha) + green.astype(np.float32) * (m3 * alpha)
        return out.clip(0, 255).astype(np.uint8)

    def auto_tune(self, bgr, target=(0.40, 0.60)):
        """基于单帧自动推导肤色判据参数，返回经钳制的参数字典。
        1) 种子: 模型皮肤类>0.5；不足时用“人脸椭圆∩宽松判据”补(剔头发/背景)；
        2) 种子像素颜色统计 → Cr/Cb/色相 中心与容差、饱和/亮度界；
        3) 迭代缩放容差使小图覆盖率逼近 target 区间(分割只跑一次，快速)；
        4) 结果经 norm_skin_params 钳制，绝不越界。"""
        if bgr is None or self.backend is None:
            return norm_skin_params()
        h, w = bgr.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(h, w))
        small = bgr if scale >= 1.0 else cv2.resize(
            bgr, (max(int(w * scale), 8), max(int(h * scale), 8)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        cr, cb = ycrcb[..., 1], ycrcb[..., 2]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # ---- 种子与分割组件(分割/检测只跑一次) ----
        seed = np.zeros((sh, sw), bool)
        base = np.zeros((sh, sw), np.float32)      # 不随参数变化的部分(模型皮肤/人脸椭圆)
        person_g = None
        not_hair_g = None
        person_raw = None
        hair_raw = None
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        det_bgr = self._det_bgr(bgr, h, w)
        fusion_bgr = self._side_bgr(bgr, h, w, self.fusion_side)
        chs_f = _channels(fusion_bgr)          # 迭代中颜色通道只算一次
        faces = self._detect_faces(cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB),
                                   sw / det_bgr.shape[1], sh / det_bgr.shape[0]) \
            if (self.backend == "mediapipe" and self._dets) else []
        if self.backend == "mediapipe":
            import mediapipe as mp
            try:
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                conf = self._seg.segment(img).confidence_masks
                if conf:
                    def _view(c):
                        v = np.asarray(conf[c].numpy_view(), np.float32)
                        return v[..., 0] if v.ndim == 3 else v
                    for c in SKIN_CLASSES:
                        if c < len(conf):
                            v = _view(c)
                            base = np.maximum(base, v)
                            seed |= v > 0.5
                    person_raw = 1.0 - _view(0)
                    person_g = np.clip(person_raw * 1.5, 0, 1)
                    if len(conf) > 1:
                        hair_raw = _view(1)
                        not_hair_g = np.clip((1.0 - hair_raw) * 1.5, 0, 1)
                ys, xs = np.ogrid[:sh, :sw]
                for cx, cy, ax, ay in faces:
                    ell = ((xs - cx) / max(ax, 1)) ** 2 + ((ys - cy) / max(ay, 1)) ** 2
                    ell_m = np.clip(1.6 - ell, 0, 1).astype(np.float32)
                    if person_g is not None:
                        ell_m = ell_m * np.clip(person_g * 1.5, 0, 1)
                    base = np.maximum(base, ell_m)
            except Exception as ex:
                self._log("[皮肤遮罩] 自动调优分割失败: " + str(ex)[:120])
        if int(seed.sum()) < 200:
            # 种子不足: 人脸椭圆 ∩ 宽松判据(交集剃掉椭圆内头发/背景)；
            # 判据在高分辨率算再下采样(小图纹理门控失效)
            if det_bgr is not None:
                loose = cv2.resize(_color_skin_chs(chs_f), (sw, sh),
                                   interpolation=cv2.INTER_AREA) > 0.2
            else:
                loose = _color_skin(small) > 0.2
            ell_m = np.zeros((sh, sw), np.float32)
            ys, xs = np.ogrid[:sh, :sw]
            for cx, cy, ax, ay in faces:
                e = ((xs - cx) / max(ax, 1)) ** 2 + ((ys - cy) / max(ay, 1)) ** 2
                ell_m = np.maximum(ell_m, (e <= 1.0).astype(np.float32))
            if not faces and self._haar is not None:
                try:
                    for (x, y, fw, fh) in self._haar.detectMultiScale(
                            small, 1.15, 5, minSize=(max(24, sw // 24),) * 2):
                        e = ((xs - (x + fw / 2)) / max(fw * 0.62, 1)) ** 2 + \
                            ((ys - (y + fh / 2)) / max(fh * 0.78, 1)) ** 2
                        ell_m = np.maximum(ell_m, (e <= 1.0).astype(np.float32))
                except Exception:
                    pass
            seed |= (ell_m > 0.5) & loose
        if int(seed.sum()) < 100 and person_g is not None:
            # 最后兜底: 人体区 ∩ 宽松肤色 ∩ 平滑纹理(检测/分割全失效时仍能采样)
            if det_bgr is not None:
                col_hi = cv2.resize(_color_skin_chs(chs_f), (sw, sh), interpolation=cv2.INTER_AREA)
                tex_hi = cv2.resize(_tex_gate(fusion_bgr, self.tex_k), (sw, sh),
                                    interpolation=cv2.INTER_AREA)
            else:
                col_hi, tex_hi = _color_skin(small), _tex_gate(small)
            seed |= (person_g > 0.5) & (col_hi > 0.15) & (tex_hi > 0.5)
        if int(seed.sum()) < 100:
            return norm_skin_params()   # 实在没种子: 保持默认

        # ---- 种子颜色统计 → 参数初值 ----
        sel = seed
        cr_c = float(cr[sel].mean())
        cb_c = float(cb[sel].mean())
        cr_w = float(np.clip(2.0 * cr[sel].std(), 8, 64))
        cb_w = float(np.clip(2.0 * cb[sel].std(), 6, 64))
        ang = np.deg2rad(hue[sel] * 2.0)                      # 色相环绕统计
        h_c = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180.0
        d_h = np.minimum(np.abs(hue[sel] - h_c), 180.0 - np.abs(hue[sel] - h_c))
        h_w = float(np.clip(2.0 * d_h.std(), 4, 45))
        s_lo = float(np.clip(sat[sel].mean() - 2.0 * sat[sel].std(), 0, 200))
        s_hi = float(np.clip(sat[sel].mean() + 2.0 * sat[sel].std() + 60.0, 100, 255))
        v_lo = float(np.clip(val[sel].mean() - 3.0 * val[sel].std(), 0, 255))
        p = norm_skin_params(dict(cr_c=cr_c, cr_w=cr_w, cb_c=cb_c, cb_w=cb_w,
                                  h_c=h_c, h_w=h_w, s_lo=s_lo, s_hi=s_hi, v_lo=v_lo))

        # ---- 迭代缩放容差逼近目标覆盖率 ----
        def cov_of(q):
            if fusion_bgr is not None and person_raw is not None:
                col = self._fusion_hi(fusion_bgr, q, person_raw, hair_raw, sw, sh, chs=chs_f)
            else:
                col = _color_skin(small, q) * _tex_gate(small)
            return float((np.maximum(base, col) > 0.35).mean())

        prev = -1.0
        for _ in range(8):
            c = cov_of(p)
            if target[0] <= c <= target[1] or abs(c - prev) < 1e-4:
                break
            prev = c
            if c < target[0]:
                p = norm_skin_params(dict(
                    p, cr_w=p['cr_w'] * 1.25, cb_w=p['cb_w'] * 1.25, h_w=p['h_w'] * 1.25,
                    s_lo=max(0.0, p['s_lo'] - 5.0), v_lo=max(0.0, p['v_lo'] - 10.0)))
            else:
                p = norm_skin_params(dict(
                    p, cr_w=p['cr_w'] * 0.85, cb_w=p['cb_w'] * 0.85, h_w=p['h_w'] * 0.85))
        return norm_skin_params(p)

    # ---------- internals ----------
    def _detect_faces(self, det_rgb, kx, ky):
        """所有人脸检测器跑 det_rgb(高分辨率通道)，返回小图坐标系的
        椭圆参数列表 [(cx, cy, ax, ay), ...]。kx/ky = 小图/检测图 缩放比。"""
        import mediapipe as mp
        out = []
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(det_rgb))
        for det in self._dets:
            try:
                for d in det.detect(img).detections:
                    bb = d.bounding_box
                    out.append(((bb.origin_x + bb.width / 2) * kx,
                                (bb.origin_y + bb.height / 2) * ky,
                                bb.width * kx * 0.62, bb.height * ky * 0.78))
            except Exception as ex:
                self._log("[皮肤遮罩] 人脸检测失败一次: " + str(ex)[:120])
        return out

    def _side_bgr(self, bgr, h, w, side):
        """按最大边 side 缩小的图(已小于则原图返回)。"""
        dscale = min(1.0, side / max(h, w))
        if dscale >= 1.0:
            return bgr
        return cv2.resize(bgr, (max(int(w * dscale), 8), max(int(h * dscale), 8)),
                          interpolation=cv2.INTER_AREA)

    def _det_bgr(self, bgr, h, w):
        """人脸检测用高分辨率图(边长 _DET_SIDE)。"""
        return self._side_bgr(bgr, h, w, _DET_SIDE)

    def _fusion_hi(self, fusion_bgr, p, person, hair, sw, sh, chs=None):
        """肤色判据融合在高分辨率通道计算(小图下纹理门控几乎全灭，
        高分辨率下平滑皮肤才能通过)，再下采样回小图。
        chs = _channels(fusion_bgr) 可预计算传入(迭代场景省颜色转换)。"""
        if chs is None:
            chs = _channels(fusion_bgr)
        fused = _color_skin_chs(chs, p) * _tex_gate(fusion_bgr, self.tex_k)
        dh, dw = fusion_bgr.shape[:2]
        if person is not None:
            person_up = cv2.resize(person, (dw, dh), interpolation=cv2.INTER_LINEAR)
            fused = fused * np.clip(person_up * 1.5, 0, 1)
        if hair is not None:
            hair_up = cv2.resize(hair, (dw, dh), interpolation=cv2.INTER_LINEAR)
            fused = fused * np.clip((1.0 - hair_up) * 1.5, 0, 1)
        return cv2.resize(fused, (sw, sh), interpolation=cv2.INTER_AREA)

    def _raw_mask(self, bgr, p):
        h, w = bgr.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(h, w))
        small = bgr if scale >= 1.0 else cv2.resize(
            bgr, (max(int(w * scale), 8), max(int(h * scale), 8)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        faces = []
        det_bgr = None
        if self.backend == "mediapipe" and self._dets:
            det_bgr = self._det_bgr(bgr, h, w)
            faces = self._detect_faces(cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB),
                                       sw / det_bgr.shape[1], sh / det_bgr.shape[0])
        if self.backend == "mediapipe":
            fusion_bgr = self._side_bgr(bgr, h, w, self.fusion_side)
            seg_views = None
            if self._seg_cache is not None and self.seg_stride > 1 and (self._frame_idx % self.seg_stride):
                seg_views = self._seg_cache      # 时域复用: 分割最贵, 相邻帧 person/hair 几乎不变
            return self._raw_mediapipe(small, rgb, sw, sh, p, faces, fusion_bgr, seg_views)
        return self._raw_opencv(small, rgb, sw, sh, p)

    def _raw_mediapipe(self, small, rgb, sw, sh, p, faces, fusion_bgr, seg_views):
        raw = np.zeros((sh, sw), np.float32)
        person = None
        hair = None
        if seg_views is not None:
            person, hair, skin = seg_views
            raw = skin.copy()
        else:
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            try:
                res = self._seg.segment(img)
                conf = res.confidence_masks
                if conf:
                    def _view(c):
                        v = np.asarray(conf[c].numpy_view(), np.float32)
                        return v[..., 0] if v.ndim == 3 else v
                    for c in SKIN_CLASSES:
                        if c < len(conf):
                            raw = np.maximum(raw, _view(c))
                    person = 1.0 - _view(0)     # 1 - background
                    if len(conf) > 1:
                        hair = _view(1)
                    if person is not None:
                        # numpy_view 是指向 C++ 结果内部缓冲的借用视图: 分割结果析构后内存即释放,
                        # 跨帧缓存必须拷贝, 否则后续帧读到已释放内存直接 AV (实测 0xC0000005)
                        self._seg_cache = (person.copy(),
                                           hair.copy() if hair is not None else None,
                                           raw.copy())
            except Exception as ex:
                self._log("[皮肤遮罩] 分割失败一次: " + str(ex)[:120])
        # 肤色判据融合：分割模型对 CG/动漫角色常把躯干皮肤误判成衣服，
        # 用“肤色判据 + 纹理门控”补全，乘 person/非头发 防背景误报；
        # 融合在高分辨率通道算(小图纹理门控失效)
        if person is not None and fusion_bgr is not None:
            raw = np.maximum(raw, self._fusion_hi(fusion_bgr, p, person, hair, sw, sh))
        # 人脸椭圆：把检测框扩成椭圆并拉到 1，补全侧脸/暗光/小脸下分割漏掉的皮肤；
        # 乘上人体区域，避免椭圆溢出到背景形成光晕
        if faces:
            ys, xs = np.ogrid[:sh, :sw]
            for cx, cy, ax, ay in faces:
                ell = ((xs - cx) / max(ax, 1)) ** 2 + ((ys - cy) / max(ay, 1)) ** 2
                ell = np.clip(1.6 - ell, 0, 1).astype(np.float32)
                if person is not None:
                    ell = ell * np.clip(person * 1.5, 0, 1)
                raw = np.maximum(raw, ell)
        return raw

    def _raw_opencv(self, small, rgb, sw, sh, p):
        # Haar 人脸（可能不存在：opencv 5.0 wheel 不带 XML）
        faces = []
        if self._haar is not None:
            try:
                faces = self._haar.detectMultiScale(
                    small, 1.15, 5, minSize=(max(24, sw // 24),) * 2)
            except Exception:
                faces = []
        face_mask = np.zeros((sh, sw), np.float32)
        for (x, y, fw, fh) in faces:
            cx, cy = x + fw / 2, y + fh / 2
            ax, ay = fw * 0.62, fh * 0.78
            ys, xs = np.ogrid[:sh, :sw]
            ell = ((xs - cx) / max(ax, 1)) ** 2 + ((ys - cy) / max(ay, 1)) ** 2
            face_mask = np.maximum(face_mask, np.clip(1.6 - ell, 0, 1).astype(np.float32))
        # 肤色判据 + 纹理门控（格纹/条纹衣物抑制）
        skin_color = _color_skin(small, p) * _tex_gate(small)
        # 有人脸时把肤色限制在"人体"范围（人脸膨胀区），降低背景误报
        if len(faces):
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            body_zone = cv2.dilate(face_mask, kern, iterations=3)
            skin_color = np.where(body_zone > 0.05, skin_color, skin_color * 0.25)
        return np.maximum(face_mask, skin_color)

    def _postprocess(self, raw, w, h):
        if raw.shape[:2] != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        m = (np.clip(raw, 0, 1) > 0.35).astype(np.float32)
        if m.max() < 1:
            return m
        k = max(3, min(w, h) // 240)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)     # 去碎点
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)    # 填小洞
        blur = max(5, (min(w, h) // 90) | 1)              # 羽化边缘
        return cv2.GaussianBlur(m, (blur, blur), 0)
