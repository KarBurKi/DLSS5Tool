#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py — 简约 DLSS5 实时预览 + 实时播放 + 导出 (test5)

功能：导入视频/图片 → 实时预览(原图/DLSS/对比) → 调风格/强度/本地色调/本地结构
      → 逐帧实时看出效果 → 实时播放(带声音) → 导出 DLSS 视频/图片。

零引导（Feature 18 神经渲染忽略光流/深度），无需 torch/模型，只需 NVIDIA 显卡。
运行： python gui.py
"""
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import queue
import types
import tkinter as tk
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk, filedialog, messagebox, scrolledtext

if getattr(sys, "frozen", False):
    # 冻结环境兼容层(必须在下方 import audio_out / skin_mask 之前):
    # 1) PyInstaller 把 sounddevice 模块级的 `import _sounddevice_data` 收进 PYZ 时，
    #    因该目录无 __init__.py 只能生成空模块(无 __path__)。sounddevice 找不到系统
    #    PortAudio 时会取 _sounddevice_data.__path__ 拼随包 DLL 路径 → AttributeError，
    #    连带 mediapipe(import 链上有 sounddevice)失败 → 皮肤遮罩静默回退 OpenCV(实测)。
    #    预置带 __path__ 的模块指向 _internal 数据目录即可修复。
    # 2) 打包时 --exclude-module matplotlib，而 mediapipe.tasks.python.vision 的
    #    drawing_utils 模块级 `import matplotlib.pyplot`(仅 import 语句本身用到，
    #    模块级无属性访问，我们也不调画图 API)。排除包时预置空壳模块让链路不断。
    _sdd = os.path.join(sys._MEIPASS, "_sounddevice_data")
    if os.path.isdir(_sdd) and "_sounddevice_data" not in sys.modules:
        _m = types.ModuleType("_sounddevice_data")
        _m.__path__ = [_sdd]
        sys.modules["_sounddevice_data"] = _m
    import importlib.util as _ilu
    if "matplotlib" not in sys.modules and _ilu.find_spec("matplotlib") is None:
        for _n in ("matplotlib", "matplotlib.pyplot"):
            sys.modules.setdefault(_n, types.ModuleType(_n))

import cv2
import numpy as np

import dlss_engine
import perf_profile

try:
    import skin_mask
    SKIN_MASK_AVAILABLE = True
except Exception:
    SKIN_MASK_AVAILABLE = False

try:
    import audio_out
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    import screen_cap
    SCREEN_CAP_AVAILABLE = True
except Exception:
    SCREEN_CAP_AVAILABLE = False

VIEWS = ["原图", "DLSS", "对比"]
STYLE_CHOICES = {"默认": 0, "自然": 1, "电影": 2, "风格3": 3}
OUTVIEW_CHOICES = {"处理": 0, "差异×10": 1, "左右对比": 2}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# 肤色判据滑条: (参数键, 标签, 下限, 上限)——默认值见 skin_mask.DEFAULT_SKIN_PARAMS
SKIN_SLIDERS = [
    ('cr_c', "Cr中心", 0.0, 255.0), ('cr_w', "Cr容差", 4.0, 64.0), ('cb_c', "Cb中心", 0.0, 255.0),
    ('cb_w', "Cb容差", 4.0, 64.0), ('h_c', "色相中心", 0.0, 180.0), ('h_w', "色相容差", 2.0, 45.0),
    ('s_lo', "饱和下界", 0.0, 200.0), ('s_hi', "饱和上界", 100.0, 255.0), ('v_lo', "亮度下界", 0.0, 255.0),
]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class App:
    def __init__(self, root):
        self.root = root
        root.title("DLSS5 简约工具 — 实时预览 + 播放 + 导出")
        root.geometry("980x830")
        self.video = None
        self.media_kind = None          # "video" | "image"
        self._img = None                # BGR array in image mode
        self._media_w = 0
        self._media_h = 0
        self.nframes = 0
        self.fps = 30.0
        self.thread = None
        self.split_x = 0.5
        self.playing = False
        self._exporting = False
        self._live = None
        self._live_cache = None
        self._last_dlss_frame = -1
        self._live_debounce = None
        self._masker = None             # skin_mask.SkinMasker 懒加载单例
        self._maskers_extra = []        # P6 导出期的额外遮罩实例(交错帧并行)
        self._dual_delta_checked = False  # 双通道差异自检只报告一次

        # ---- realtime playback state ----
        self._pb_thread = None
        self._pb_lock = threading.Lock()
        self._pb_latest = None          # (idx, orig_small_bgr, dlss_small_bgr)
        self._pb_stop = threading.Event()
        self._pb_seek_req = None
        self._pb_settings_pending = None
        self._pb_ui_suppress = False
        self._pb_lag_logged = False
        self._pb_after = None
        self._pb_anchor_frame = 0
        self._pb_anchor_wall = 0.0
        self._pb_audio_reanchored = False
        self._pb_draw_key = None
        self._pb_speed = 1.0
        self._has_audio = False
        self._audio = audio_out.AudioPlayer(log=self._log_safe) if AUDIO_AVAILABLE else None

        # ---- realtime screen-capture test state ----
        self._sc_active = False
        self._sc_thread = None
        self._sc_lock = threading.Lock()
        self._sc_latest = None          # (orig_small_bgr, dlss_small_bgr)
        self._sc_count = 0              # 已发布帧计数（绘制去重键）
        self._sc_stop = threading.Event()
        self._sc_settings_pending = None
        self._sc_after = None
        self._sc_draw_key = None
        self._sc_fps = 0.0
        self._sc_cap = None
        self._sc_region = None            # 框选区域 (l, t, r, b)，绝对屏幕坐标
        self._sc_monitors = screen_cap.list_monitors() if SCREEN_CAP_AVAILABLE else []

        # ---- video row ----
        t = ttk.Frame(root); t.pack(fill="x", padx=8, pady=6)
        self.import_btn = ttk.Button(t, text="导入视频/图片", command=self.import_media)
        self.import_btn.pack(side="left")
        self._topmost = False
        self.top_btn = ttk.Button(t, text="📌 置顶", width=9, command=self.on_toggle_topmost)
        self.top_btn.pack(side="left", padx=(8, 0))
        self.vlabel = ttk.Label(t, text="(未选择视频/图片)", anchor="w")
        self.vlabel.pack(side="left", padx=8, fill="x", expand=True)

        # ---- preview canvas (aspect ratio preserved), default view = 对比 ----
        self.canvas = tk.Canvas(root, bg="#161616", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>", lambda e: self.display_view())
        self.canvas.bind("<B1-Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-1>", self.on_canvas_motion)

        # ---- timeline / scrubbing bar (BELOW the video) ----
        self.fslider = tk.Scale(root, from_=0, to=1, orient="horizontal", showvalue=False,
                                resolution=1, command=lambda e: self.on_frame())
        self.fslider.pack(fill="x", padx=8, pady=(0, 2))
        self.fslider.bind("<Button-1>", self.on_timeline_click)
        self.fslider.bind("<B1-Motion>", self.on_timeline_drag)

        # ---- view + frame + playback (BELOW the video, under the timeline) ----
        v = ttk.Frame(root); v.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(v, text="显示:").pack(side="left")
        self.view_var = tk.StringVar(value="对比")
        self.view_cb = ttk.Combobox(v, textvariable=self.view_var, values=VIEWS, state="readonly", width=7)
        self.view_cb.pack(side="left", padx=3)
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: self.on_view_change())
        ttk.Label(v, text="  帧:").pack(side="left")
        self.fentry = tk.Entry(v, width=7)
        self.fentry.pack(side="left", padx=3)
        self.fentry.insert(0, "0")
        self.fentry.bind("<Return>", self.on_frame_entry)
        self.fentry.bind("<FocusOut>", lambda e: self.sync_frame_entry())
        self.ftotal = ttk.Label(v, text="/ 0")
        self.ftotal.pack(side="left", padx=(0, 6))
        self.play_btn = ttk.Button(v, text="▶ 播放", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=(8, 0))
        ttk.Label(v, text="  音量:").pack(side="left", padx=(10, 2))
        self.vol_var = tk.DoubleVar(value=1.0)
        self._slider_entry(v, self.vol_var, self.on_volume_change, length=100, range_max=2.0).pack(side="left")
        ttk.Label(v, text="  速度:").pack(side="left", padx=(10, 2))
        self.speed_var = tk.StringVar(value="1.0")
        self.speed_cb = ttk.Combobox(v, textvariable=self.speed_var,
                                     values=["0.25", "0.5", "0.75", "1.0"],
                                     state="readonly", width=5)
        self.speed_cb.pack(side="left")
        self.speed_cb.bind("<<ComboboxSelected>>", lambda e: self.on_speed_change())

        # ---- DLSS settings ----
        sf = ttk.LabelFrame(root, text="DLSS 设置")
        sf.pack(fill="x", padx=8, pady=4)
        self._settings = self._build_settings(sf)

        # ---- skin-color criteria (mask fine-tuning) ----
        self._skin_vars = {}
        kf = ttk.LabelFrame(root, text="肤色判据(皮肤遮罩微调)")
        kf.pack(fill="x", padx=8, pady=4)
        self._build_skin_params(kf)

        # ---- output (export) section ----
        of = ttk.LabelFrame(root, text="输出")
        of.pack(fill="x", padx=8, pady=4)
        self.v_outview = tk.StringVar(value="处理")
        ttk.Label(of, text="输出视图:").pack(side="left", padx=(6, 2))
        self.outview_cb = ttk.Combobox(of, textvariable=self.v_outview, values=list(OUTVIEW_CHOICES), state="readonly", width=9)
        self.outview_cb.pack(side="left", padx=(0, 8))
        ttk.Label(of, text="(导出时按当前 DLSS 设置数值生成)").pack(side="left", padx=(0, 12))
        self.export_btn = ttk.Button(of, text="导出 DLSS 视频", command=self.export_dlss)
        self.export_btn.pack(side="left", padx=3)
        ttk.Label(of, text="  导出质量(CRF 0-51，越小越清晰，默认18):").pack(side="left", padx=(8, 2))
        self.crf = tk.IntVar(value=18)
        self.crf_spin = ttk.Spinbox(of, from_=0, to=51, textvariable=self.crf, width=6)
        self.crf_spin.pack(side="left")

        # ---- realtime screen-capture test ----
        cf = ttk.LabelFrame(root, text="实时截屏测试")
        cf.pack(fill="x", padx=8, pady=4)
        ttk.Label(cf, text="显示器:").pack(side="left", padx=(6, 2))
        mon_names = ["%s (%dx%d)" % (m["device"], m["rect"][2] - m["rect"][0],
                                     m["rect"][3] - m["rect"][1])
                     for m in self._sc_monitors]
        self.sc_mon_var = tk.StringVar(value=mon_names[0] if mon_names else "(未检测到)")
        self.sc_mon_cb = ttk.Combobox(cf, textvariable=self.sc_mon_var, values=mon_names,
                                      state="readonly", width=26)
        self.sc_mon_cb.pack(side="left", padx=(0, 8))
        self.sc_pick_btn = ttk.Button(cf, text="⬚ 框选区域", command=self._sc_pick_region)
        self.sc_pick_btn.pack(side="left", padx=(0, 8))
        self.sc_btn = ttk.Button(cf, text="▶ 开始截屏测试", command=self._sc_toggle)
        self.sc_btn.pack(side="left", padx=(0, 8))
        self.sc_region_label = ttk.Label(cf, text="未选区域")
        self.sc_region_label.pack(side="left", padx=(0, 8))
        self.sc_fps_label = ttk.Label(cf, text="")
        self.sc_fps_label.pack(side="left", padx=(0, 12))
        ttk.Label(cf, text="(先框选区域再开始；结果显示在上方预览区)").pack(side="left")
        if not SCREEN_CAP_AVAILABLE:
            self.sc_btn.config(state="disabled")
            self.sc_pick_btn.config(state="disabled")

        # ---- progress ----
        p = ttk.Frame(root); p.pack(fill="x", padx=8, pady=2)
        self.pbar = ttk.Progressbar(p, maximum=100)
        self.pbar.pack(fill="x", expand=True)
        self.status = ttk.Label(p, text="就绪")
        self.status.pack(fill="x")

        # ---- log ----
        self.log = scrolledtext.ScrolledText(root, height=7, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=False, padx=8, pady=4)

        # ---- spacebar toggles play/pause (anywhere in the window, except text fields) ----
        self.root.bind_all("<space>", self.on_space)

        # 工作线程→UI 的中转: 线程只写属性/列表, 主线程轮询消费 (P1)
        self._pending_logs = []
        self._export_prog = None
        self._export_done = None     # (out_path, ok) 完成后由轮询弹窗
        self._pb_btn_pending = False  # 播放线程请求复位播放按钮
        self._cov_pending = None      # 覆盖率文本待刷新(工作线程写)
        self._reset_skin_pending = False
        self._sc_stop_pending = False
        self.root.after(120, self._ui_poll)

        # 预览绘制依赖 Pillow(PIL): 启动时检测一次并提示, 避免每帧刷同一条英文异常
        self._pil_ok = True
        try:
            from PIL import Image, ImageTk  # noqa: F401
        except Exception as ex:
            self._pil_ok = False
            self.logln("[preview] 预览需要 Pillow：python -m pip install pillow")
            self.logln("[preview] 缺失原因: %s (预览区不可用, 导出/播放声音等功能不受影响)" % ex)
            self.set_status("预览需要 Pillow：python -m pip install pillow")

    def _ui_poll(self):
        """主线程轮询: 消费工作线程的日志/导出进度/导出完成事件。"""
        try:
            while self._pending_logs:
                self.logln(self._pending_logs.pop(0))
            ep = self._export_prog
            if ep is not None:
                self._export_prog = None
                self.set_progress(ep[0], ep[1], "导出")
            ed = self._export_done
            if ed is not None:
                self._export_done = None
                self._export_finish(ed[0], ed[1])
            if self._pb_btn_pending:
                self._pb_btn_pending = False
                self._set_play_btn(False)
            cv_txt = self._cov_pending
            if cv_txt is not None:
                self._cov_pending = None
                lbl = getattr(self, "_cov_lbl", None)
                if lbl is not None:
                    lbl.config(text=cv_txt)
            if self._reset_skin_pending:
                self._reset_skin_pending = False
                self._reset_skin_params()
            if self._sc_stop_pending:
                self._sc_stop_pending = False
                self._sc_stop_capture()
        except Exception:
            pass
        self.root.after(120, self._ui_poll)

    # ---------- helpers ----------
    def set_status(self, msg):
        try:
            self.status.config(text=msg); self.root.update_idletasks()
        except Exception:
            pass

    def logln(self, msg):
        try:
            self.log.config(state="normal")
            self.log.insert("end", msg + "\n"); self.log.see("end")
            self.log.config(state="disabled")
        except Exception:
            pass

    def _log_safe(self, msg):
        # thread-safe log from worker/audio threads (root.after 从非主线程调会抛
        # "main thread is not in main loop", 改用属性 + 主线程轮询)
        try:
            self._pending_logs.append(msg)
        except Exception:
            pass

    def set_progress(self, i, total, extra=""):
        try:
            if total:
                self.pbar["maximum"] = total
                self.pbar["value"] = i
                self.set_status(f"{extra} {i}/{total}")
            self.root.update_idletasks()
        except Exception:
            pass

    def _read_frame(self, frame):
        if self.media_kind == "image":
            return self._img
        cap = getattr(self, "_cap", None)
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, f = cap.read()
        return f if ok else None

    def _worker_alive(self):
        return self._pb_thread is not None and self._pb_thread.is_alive()

    # ---------- preview ----------
    def _slider_entry(self, parent, var, cmd, length=100, range_max=10.0, range_min=0.0):
        """滑条 + 可手动输入的数值框：拖动步进 0.01；数值框输入后回车/失去焦点生效，
        超出范围自动钳制。变量被外部修改时数值框同步刷新。"""
        f = ttk.Frame(parent)
        sc = tk.Scale(f, from_=range_min, to=range_max, resolution=0.01, orient="horizontal",
                      showvalue=False, variable=var, length=length, command=cmd)
        sc.pack(side="left")
        en = tk.Entry(f, width=6, justify="right")
        en.pack(side="left", padx=(3, 0))

        def var_to_entry(*_):
            try:
                en.delete(0, "end")
                en.insert(0, "%.2f" % float(var.get()))
            except Exception:
                pass

        def commit(_=None):
            try:
                v = float(en.get().strip())
            except ValueError:
                var_to_entry()
                return
            v = max(float(sc.cget("from")), min(float(sc.cget("to")), v))
            var.set(v)
            cmd(None)

        var.trace_add("write", var_to_entry)
        en.bind("<Return>", commit)
        en.bind("<FocusOut>", commit)
        var_to_entry()
        f._commit = commit
        return f

    def _build_settings(self, parent):
        d = {}
        d['v_style'] = tk.StringVar(value="默认")
        d['v_intensity'] = tk.DoubleVar(value=1.0)
        d['v_local_tone'] = tk.DoubleVar(value=1.0)
        d['v_local_struct'] = tk.DoubleVar(value=1.0)
        d['v_skin_struct'] = tk.DoubleVar(value=1.0)
        cb_style = ttk.Combobox(parent, textvariable=d['v_style'], values=list(STYLE_CHOICES), state="readonly", width=8)
        cb_style.bind("<<ComboboxSelected>>", lambda e: self.on_settings_change())
        ttk.Label(parent, text="风格:").grid(row=0, column=0, sticky="e", padx=(6, 2), pady=2)
        cb_style.grid(row=0, column=1, padx=(0, 8))
        ttk.Label(parent, text="强度:").grid(row=0, column=2, sticky="e", padx=(6, 2))
        self._slider_entry(parent, d['v_intensity'], self.on_settings_change, length=100,
                           range_min=0.0, range_max=5.0).grid(row=0, column=3, padx=(0, 8))
        ttk.Label(parent, text="本地色调:").grid(row=0, column=4, sticky="e", padx=(6, 2))
        self._slider_entry(parent, d['v_local_tone'], self.on_settings_change, length=90,
                           range_min=-2.0, range_max=5.0).grid(row=0, column=5, padx=(0, 8))
        ttk.Label(parent, text="本地结构:").grid(row=0, column=6, sticky="e", padx=(6, 2))
        self._slider_entry(parent, d['v_local_struct'], self.on_settings_change, length=90,
                           range_min=-2.0, range_max=5.0).grid(row=0, column=7, padx=(0, 8))
        ttk.Label(parent, text="皮肤结构:").grid(row=1, column=0, sticky="e", padx=(6, 2))
        self._slider_entry(parent, d['v_skin_struct'], self.on_settings_change, length=100,
                           range_min=-2.0, range_max=5.0).grid(row=1, column=1, padx=(0, 8))
        d['v_skin_dual'] = tk.BooleanVar(value=True)
        d['v_show_mask'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="皮肤双通道(遮罩混合)",
                        variable=d['v_skin_dual'], command=self.on_settings_change).grid(row=1, column=2, columnspan=2, sticky="w", padx=(6, 2))
        ttk.Checkbutton(parent, text="叠加显示遮罩",
                        variable=d['v_show_mask'], command=self.on_settings_change).grid(row=1, column=4, columnspan=2, sticky="w", padx=(2, 2))
        ttk.Label(parent, text="(风格会实时生效；强度范围 0~5.00，本地色调/本地结构/皮肤结构范围 -2.00~5.00；数值框可手动输入，回车生效，超范围自动钳)").grid(row=2, column=0, columnspan=8, sticky="w", padx=(6, 2))
        return d

    def _build_skin_params(self, parent):
        """肤色判据滑条组：参数进 settings['skin_params']，拖动后实时重算遮罩，
        覆盖率标签反馈调整效果。skin_mask 不可用时仅提示。"""
        self._cov_last = 0.0
        if not SKIN_MASK_AVAILABLE:
            ttk.Label(parent, text="(skin_mask 不可用，肤色判据调节已禁用)").grid(
                row=0, column=0, sticky="w", padx=(6, 2))
            return
        for i, (key, label, lo, hi) in enumerate(SKIN_SLIDERS):
            r, c = divmod(i, 3)
            self._skin_vars[key] = tk.DoubleVar(value=skin_mask.DEFAULT_SKIN_PARAMS[key])
            ttk.Label(parent, text=label + ":").grid(row=r, column=c * 2, sticky="e", padx=(6, 2), pady=1)
            self._slider_entry(parent, self._skin_vars[key], self.on_settings_change,
                               length=80, range_min=lo, range_max=hi).grid(row=r, column=c * 2 + 1, padx=(0, 6))
        self._cov_lbl = ttk.Label(parent, text="皮肤遮罩覆盖率: --")
        self._cov_lbl.grid(row=3, column=0, columnspan=3, sticky="w", padx=(6, 2))
        ttk.Button(parent, text="恢复默认", command=self._reset_skin_params).grid(row=3, column=3, sticky="w", padx=(2, 6))
        ttk.Button(parent, text="一键自动调优", command=self._auto_tune_skin).grid(row=3, column=4, sticky="w", padx=(2, 6))
        ttk.Label(parent, text="(拖动实时重算遮罩，配合“叠加显示遮罩”观察；OpenCV 后端下极端值使遮罩失效时自动恢复默认)").grid(
            row=4, column=0, columnspan=6, sticky="w", padx=(6, 2))

    def _reset_skin_params(self):
        """主线程: 肤色判据全部恢复默认并触发重渲染。"""
        if not SKIN_MASK_AVAILABLE:
            return
        for k, v in self._skin_vars.items():
            v.set(skin_mask.DEFAULT_SKIN_PARAMS[k])
        self.on_settings_change()

    def _current_bgr(self):
        """当前预览帧 BGR (自动调优/调试用)。"""
        if getattr(self, "media_kind", None) == "image":
            return getattr(self, "_img", None)
        if getattr(self, "nframes", 0) <= 0:
            return None
        try:
            idx = int(float(self.fslider.get()))
        except Exception:
            idx = 0
        return self._read_frame(max(0, min(idx, self.nframes - 1)))

    def _auto_tune_skin(self):
        """一键自动调优: 基于当前帧颜色统计推导肤色判据参数并写回滑条，
        立即重算遮罩并刷新覆盖率。"""
        if not SKIN_MASK_AVAILABLE:
            return
        masker = self._ensure_masker()
        if masker is None or not masker.backend:
            self.logln("[肤色判据] 自动调优失败: 遮罩后端不可用")
            return
        bgr = self._current_bgr()
        if bgr is None:
            self.logln("[肤色判据] 自动调优失败: 未导入素材")
            return
        p = masker.auto_tune(bgr)
        for k, v in self._skin_vars.items():
            v.set(p[k])
        m = masker.mask(bgr, params=p)
        cov = float((m > 0.5).mean()) * 100.0
        self._cov_safe(cov, force=True)
        self.logln("[肤色判据] 自动调优完成: Cr中心=%.0f, Cb中心=%.0f, 覆盖率=%.1f%%"
                   % (p['cr_c'], p['cb_c'], cov))
        self.on_settings_change()

    def _reset_skin_params_safe(self):
        self._reset_skin_pending = True

    def _cov_safe(self, pct, force=False):
        """线程安全的覆盖率标签更新，0.25s 节流。"""
        lbl = getattr(self, "_cov_lbl", None)
        if lbl is None:
            return
        now = time.monotonic()
        if not force and now - self._cov_last < 0.25:
            return
        self._cov_last = now
        self._cov_pending = "皮肤遮罩覆盖率: %.1f%%" % pct

    def _ensure_perf_tier(self):
        """首次使用时检测 GPU 档位并打日志 (P4)。返回 tier。"""
        if getattr(self, "_gpu_tier", None) is not None:
            return self._gpu_tier
        self._gpu_tier, self._gpu_desc = perf_profile.detect_gpu()
        self._gov_tier = self._gpu_tier
        self._log_safe("[性能] GPU 档位: %s (%s)" % (
            perf_profile.TIER_NAME[self._gpu_tier], self._gpu_desc))
        if self._gpu_tier == "none":
            self._log_safe("[性能] 未检测到 NVIDIA GPU/驱动: DLSS 不可用，首次使用时将切 CPU 回退渲染")
        else:
            # 按 GPU 系列挑 nvngx_dlssnr.dll(30/40/50系)。必须在首次 NGX init 前完成,
            # _ensure_perf_tier 在 _ensure_live 之前被调, 时机安全。冻结模式下系列
            # 附加包放在 exe 旁边(_MEIPASS 是临时目录), 一并搜索。
            exe_dir = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else None
            dll, series = perf_profile.nvngx_dll_for(self._gpu_desc, dlss_engine.BASE, exe_dir=exe_dir)
            if dlss_engine.set_plugin_dll(dll):
                self._log_safe("[性能] DLSS5 渲染库: %s (%s)" % (dll, ("按" + series + "选择") if series else "自带默认"))
            else:
                self._log_safe("[性能] DLSS5 渲染库: %s (已初始化, 不再切换)" % dlss_engine.DLSSNR_DLL)
            msg = "GPU 档位: %s | 融合=%d 遮罩步长=%d" % (
                perf_profile.TIER_NAME[self._gpu_tier],
                perf_profile.PROFILES[self._gpu_tier]['fusion_side'],
                perf_profile.PROFILES[self._gpu_tier]['raw_stride'])
            if threading.current_thread() is threading.main_thread():
                self.set_status(msg)
        return self._gpu_tier

    def _apply_perf_profile(self):
        """把当前治理器档位的 profile 应用到遮罩器(运行时可升降档)。
        包括 P6 导出期的额外实例; 预览路径用原始 profile(EMA 仍在实例内部做)。"""
        tier = getattr(self, "_gov_tier", None) or getattr(self, "_gpu_tier", "mid")
        prof = perf_profile.PROFILES.get(tier, perf_profile.PROFILES["mid"])
        for mk in [self._masker] + list(self._maskers_extra):
            if mk is not None:
                mk.apply_profile(prof)

    def _mask_with_fallback(self, masker, bgr_src, params):
        """计算皮肤遮罩并更新覆盖率；OpenCV 后端下若极端参数使遮罩失效，
        警告并自动恢复默认参数重算。"""
        m = masker.mask(bgr_src, params=params)
        self._cov_safe(float((m > 0.5).mean()) * 100.0)
        if masker.backend == 'opencv' and float(m.max()) < 0.05 and params:
            bad = any(abs(float(params.get(k, dv)) - dv) > 1e-6
                      for k, dv in skin_mask.DEFAULT_SKIN_PARAMS.items())
            if bad:
                self._log_safe("[肤色判据] 警告: 极端参数使遮罩失效(OpenCV 后端)，已恢复默认值")
                self._reset_skin_params_safe()
                m = masker.mask(bgr_src, params=None)
                self._cov_safe(float((m > 0.5).mean()) * 100.0, force=True)
        return m

    def _collect_settings(self):
        d = self._settings
        return {
            'style': STYLE_CHOICES.get(d['v_style'].get(), 0),
            'intensity': float(d['v_intensity'].get()),
            'local_tone': float(d['v_local_tone'].get()),
            'local_struct': float(d['v_local_struct'].get()),
            'skin_struct': float(d['v_skin_struct'].get()),
            'skin_dual': bool(d['v_skin_dual'].get()),
            'show_mask': bool(d['v_show_mask'].get()),
            'output_view': OUTVIEW_CHOICES.get(self.v_outview.get(), 0),
            'skin_params': {k: float(v.get()) for k, v in self._skin_vars.items()},
        }

    def _settings_hash(self):
        s = self._collect_settings()
        return (s['style'], s['intensity'], s['local_tone'], s['local_struct'],
                s['skin_struct'], s['skin_dual'], s['show_mask'],
                tuple(round(s['skin_params'].get(k, 0.0), 2) for k, _, _, _ in SKIN_SLIDERS))

    def _ensure_live(self, w, h, settings=None):
        """Reuse ONE Live session for preview + playback + export. On a size change (new
        media with a different resolution, or playback downscale) call resize() to
        re-create the Feature 18 WITHOUT re-running the NGX core init (which is one-time
        per process and crashes with an access violation)."""
        try:
            if settings is None:
                settings = self._collect_settings()
            self._ensure_perf_tier()
            need = (self._live is None) or (getattr(self, "_live_w", -1) != w) or (getattr(self, "_live_h", -1) != h)
            if need:
                if self._live:
                    self._live.resize(w, h, int(settings.get('preset', 1)))
                else:
                    self._live = dlss_engine.Live(w, h, settings)
                self._live_w, self._live_h = w, h
                self._last_dlss_frame = -1
                self._live_cache = None
                self._split_frame = -1
            else:
                self._live.update(settings)
            return self._live
        except Exception as ex:
            self._log_safe("[DLSS] " + str(ex))
            # P4 兼容回退: D3D12/驱动不可用时降级纯 CPU 渲染管线(OpenCV)
            if not getattr(self, "_cpu_fallback_warned", False):
                self._cpu_fallback_warned = True
                self._log_safe("[性能] DLSS 引擎不可用，已切换 CPU 回退渲染: 画质上限显著低于 Feature 18，速度受 CPU 限制")
            try:
                if not isinstance(self._live, dlss_engine.CpuLive):
                    self._live = dlss_engine.CpuLive(w, h, settings)
                else:
                    self._live.resize(w, h)
                    self._live.update(settings or {})
                self._live_w, self._live_h = w, h
                self._last_dlss_frame = -1
                self._live_cache = None
                self.set_status("CPU 回退模式(无 DLSS): 预期速度受 CPU 限制")
                return self._live
            except Exception:
                return None

    def _close_live(self):
        if self._live:
            try:
                self._live.close()
            except Exception:
                pass
            self._live = None
        self._live_cache = None
        self._last_dlss_frame = -1

    def _live_dlss_image(self, frame):
        if self._worker_alive():
            return None
        sk = self._settings_hash()
        if self._live_cache and self._live_cache[0] == frame and self._live_cache[1] == sk:
            return self._live_cache[2]
        fr = self._read_frame(frame)
        if fr is None:
            return None
        h, w = fr.shape[:2]
        live = self._ensure_live(w, h)
        if live is None:
            return None
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
        reset = 0 if frame == self._last_dlss_frame + 1 else 1
        if reset and self._masker is not None:
            self._masker.reset()
        o = self._process_dlss(live, rgba, fr, bool(reset), self._collect_settings())
        self._last_dlss_frame = frame
        if o is None:
            self._live_cache = None
            return None
        bgr = cv2.cvtColor(o[..., :3], cv2.COLOR_RGB2BGR)
        self._live_cache = (frame, sk, bgr)
        return bgr

    def load_view_img(self, view, frame):
        if view == "原图":
            return self._read_frame(frame)
        if view == "DLSS":
            return self._live_dlss_image(frame)
        return None

    def display_view(self):
        if self._sc_active:
            self._draw_sc_frame()
            return
        if not self.video or getattr(self, "_exporting", False):
            return
        if self.playing:
            self._draw_playback_frame()
            return
        frame = int(self.fslider.get())
        view = self.view_var.get()
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width() or 780, 200)
        ch = max(self.canvas.winfo_height() or 400, 150)
        if view == "对比":
            self._draw_split(frame, cw, ch); return
        img = self.load_view_img(view, frame)
        if img is None:
            msg = f"{view}：帧 {frame} 读取失败" if view == "原图" else f"DLSS：帧 {frame} 生成失败"
            self.canvas.create_text(cw // 2, ch // 2, text=msg, fill="#888888", font=("Microsoft YaHei", 11))
            return
        self._draw_fit(img, cw, ch)

    def _draw_fit(self, img, cw, ch):
        if not self._pil_ok:
            self._draw_pil_hint(cw, ch); return
        ih, iw = img.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
        nimg = cv2.resize(img, (nw, nh))
        from PIL import Image, ImageTk
        self._pilimg = Image.fromarray(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(self._pilimg)
        self.canvas.delete("all")
        self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor="nw", image=self._photo)

    def _draw_split(self, frame, cw, ch):
        if getattr(self, "_split_frame", -1) != frame or getattr(self, "_split_size", None) != (cw, ch):
            orig = self.load_view_img("原图", frame)
            dlss = self.load_view_img("DLSS", frame)
            if orig is None:
                self.canvas.create_text(cw // 2, ch // 2, text=f"帧 {frame} 读取失败", fill="#888"); return
            if dlss is None:
                self._draw_fit(orig, cw, ch)
                self.canvas.create_text(cw // 2, 16, text="DLSS 生成失败", fill="#888"); return
            self._split_orig = orig
            self._split_dlss = dlss
            self._split_frame = frame; self._split_size = (cw, ch)
        self._compose_split(self._split_orig, self._split_dlss, cw, ch)

    def _draw_pil_hint(self, cw, ch):
        """缺 Pillow 时在预览区画静态提示, 而不是每帧抛异常。"""
        try:
            self.canvas.delete("all")
            self.canvas.create_text(cw // 2, ch // 2 - 12, text="预览不可用：缺少 Pillow",
                                    fill="#cc6666", font=("Microsoft YaHei", 12))
            self.canvas.create_text(cw // 2, ch // 2 + 14,
                                    text="请在命令行执行:  python -m pip install pillow  后重开本工具",
                                    fill="#888888", font=("Microsoft YaHei", 10))
        except Exception:
            pass

    def _compose_split(self, orig, dlss, cw, ch):
        """Compose the 对比 view from two same-or-different-size BGR frames."""
        if not self._pil_ok:
            self._draw_pil_hint(cw, ch); return
        ih, iw = orig.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
        o = cv2.resize(orig, (nw, nh))
        d = cv2.resize(dlss, (nw, nh))
        sx = int(self.split_x * nw)
        o[:, sx:] = d[:, sx:]
        o[:, max(sx - 1, 0):min(sx + 1, nw)] = [0, 255, 255]
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self._drag_nw = nw; self._drag_offsetx = ox
        from PIL import Image, ImageTk
        self._pilimg = Image.fromarray(cv2.cvtColor(o, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(self._pilimg)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)

    def on_canvas_motion(self, event):
        if self.view_var.get() != "对比":
            return
        if not hasattr(self, "_drag_nw") or not hasattr(self, "_drag_offsetx"):
            return
        frac = (event.x - self._drag_offsetx) / max(self._drag_nw, 1)
        self.split_x = max(0.0, min(1.0, frac))
        self.display_view()

    # ---------- frame / view ----------
    def on_frame(self):
        if self._pb_ui_suppress or self.playing or self._sc_active:
            return
        self.sync_frame_entry()
        self.display_view()

    def _timeline_value_from_x(self, x):
        f = int(self.fslider.cget("from")); t = int(self.fslider.cget("to"))
        if t <= f:
            return f
        try:
            x0 = self.fslider.coords(f)[0]
            x1 = self.fslider.coords(t)[0]
        except Exception:
            x0, x1 = 10.0, max(self.fslider.winfo_width() - 10.0, 11.0)
        span = x1 - x0
        if span <= 0:
            return f
        frac = max(0.0, min(1.0, (x - x0) / span))
        return int(round(f + frac * (t - f)))

    def on_timeline_click(self, event):
        if self.playing:
            self.pause()
        self.fslider.set(self._timeline_value_from_x(event.x))
        self.on_frame()

    def on_timeline_drag(self, event):
        self.on_timeline_click(event)

    def on_frame_entry(self, event=None):
        if self.playing:
            self.pause()
        txt = self.fentry.get().strip()
        try:
            f = int(float(txt))
        except ValueError:
            f = None
        if f is None:
            self.sync_frame_entry(); return
        to = int(self.fslider.cget("to"))
        f = max(0, min(to, f))
        self.fslider.set(f)
        self.on_frame()

    def sync_frame_entry(self):
        try:
            txt = str(int(self.fslider.get()))
            if self.fentry.get().strip() != txt:
                self.fentry.delete(0, "end")
                self.fentry.insert(0, txt)
        except Exception:
            pass

    def on_view_change(self):
        if self.view_var.get() == "对比":
            self.split_x = 0.5
        self.display_view()

    def on_settings_change(self, event=None):
        if self._live_debounce:
            self.root.after_cancel(self._live_debounce)
        self._live_debounce = self.root.after(60, self._refresh_dlss)

    def on_volume_change(self, event=None):
        if self._audio is not None:
            self._audio.set_volume(float(self.vol_var.get()))

    def on_toggle_topmost(self):
        self._topmost = not self._topmost
        try:
            self.root.attributes("-topmost", self._topmost)
            self.top_btn.config(text="📌 已置顶" if self._topmost else "📌 置顶")
        except Exception:
            pass

    def on_speed_change(self, event=None):
        """播放中变速：从当前播放头用新速度重启播放（新分辨率档位随之生效）。"""
        if getattr(self, "_exporting", False) or self._sc_active:
            return
        try:
            speed = float(self.speed_var.get())
        except Exception:
            return
        if speed <= 0 or not self.playing:
            return
        pos = max(0, min(self._pb_now_frame(), max(self.nframes - 1, 0)))
        self.pause()
        self._pb_ui_suppress = True
        try:
            self.fslider.set(pos)
        finally:
            self._pb_ui_suppress = False
        self.sync_frame_entry()
        self.logln(f"[播放] 速度切换为 {speed:.2f}x，从第 {pos} 帧继续")
        self.play()

    def _refresh_dlss(self):
        self._live_debounce = None
        if self._sc_active:
            # Live is owned by the capture worker — queue the settings for it
            self._sc_settings_pending = self._collect_settings()
            return
        if self.playing:
            # Live is owned by the playback worker — queue the settings for it
            self._pb_settings_pending = self._collect_settings()
            return
        if getattr(self, "_exporting", False):
            return
        if self._live:
            try:
                self._live.update(self._collect_settings())
            except Exception as ex:
                self.logln("[DLSS 参数] " + str(ex))
        self._live_cache = None
        self._split_frame = -1
        if self.view_var.get() in ("DLSS", "对比"):
            self.display_view()

    # ---------- 皮肤双通道 (dual-pass skin masking) ----------
    def _ensure_masker(self):
        """懒加载皮肤遮罩提取器（整个进程一个实例），创建时按 GPU 档位应用性能 profile。"""
        if self._masker is None and SKIN_MASK_AVAILABLE:
            try:
                tier = self._ensure_perf_tier()
                self._masker = skin_mask.SkinMasker(log=self._log_safe)
                self._masker.apply_profile(perf_profile.PROFILES.get(tier, perf_profile.PROFILES["mid"]))
            except Exception as ex:
                self._log_safe("[皮肤遮罩] 初始化失败: " + str(ex))
        return self._masker

    def _ensure_maskers(self, k):
        """确保有 k 个遮罩实例(P6 交错帧并行用), 返回实例列表(首个为主实例)。

        每个实例自带 tflite 解释器, 约 +130MB 内存, 只在导出期按需创建, 之后一直
        缓存复用(见 _park_extra_maskers: close() 代价太大, 不关)。创建失败时自动
        降到已成功的路数, k=1 就是原来的单实例行为。"""
        first = self._ensure_masker()
        if first is None or not first.backend:
            return []
        lst = [first] + [m for m in self._maskers_extra if m is not None]
        while len(lst) < k:
            try:
                mk = skin_mask.SkinMasker(log=lambda m: None)
            except Exception as ex:
                self._log_safe("[皮肤遮罩] 第 %d 路并行实例创建失败, 降到 %d 路: %s"
                               % (len(lst) + 1, len(lst), str(ex)[:80]))
                break
            if not mk.backend:
                break
            self._maskers_extra.append(mk)
            lst.append(mk)
        if len(lst) > 1:
            self._log_safe("[性能] 遮罩并行 %d 路(交错帧; EMA 在有序点统一做, 输出与串行一致)" % len(lst))
        return lst

    def _park_extra_maskers(self):
        """导出结束: 额外遮罩实例只清时域状态、留着复用, 不 close()。

        实测 mediapipe 任务对象的 close() 要等约 100s 才返回(分割器 42s + 两个人脸
        检测器 42s/17s), 期间几乎不吃 CPU(0.04 核, 是干等), 但同进程的 mask() 会掉到
        2.2 倍耗时。所以: 同步关会让"导出完成"硬生生晚 100s, 扔后台线程则会拖慢紧接着
        的下一次导出 —— 两条路都比留着更差。留着复用还省掉每实例约 600ms 初始化,
        代价是每实例约 130MB 常驻(和主实例一样, 进程退出时才还)。
        """
        for mk in self._maskers_extra:
            try:
                mk.reset()
            except Exception:
                pass

    def _dual_buf(self, rgba):
        """双通道 A 通道的独立输出缓冲(按尺寸缓存复用)。
        必需: live.process 的内部缓冲会被 B 通道覆盖, A、B 必须分开存。"""
        buf = getattr(self, "_dual_a_buf", None)
        if buf is None or buf.shape != rgba.shape or buf.dtype != rgba.dtype:
            buf = np.empty_like(rgba)
            self._dual_a_buf = buf
        return buf

    @staticmethod
    def _blend_mask(a, b, mask):
        """按遮罩把 B 通道混进 A(原地写 a 并返回)。只在遮罩非零的包围盒内、
        只混 RGB 三通道: 皮肤覆盖率通常只有百分之几, 全画幅 float32 混合把 98%
        的算力花在必然等于 A 的像素上(720p 实测 66ms -> 9ms, 最大误差 1/255)。"""
        ys = np.flatnonzero(mask.max(axis=1) > 0.004)
        xs = np.flatnonzero(mask.max(axis=0) > 0.004)
        if ys.size == 0 or xs.size == 0:
            return a
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        m3 = mask[y0:y1, x0:x1, None]
        sa = a[y0:y1, x0:x1, :3].astype(np.float32)
        sb = b[y0:y1, x0:x1, :3].astype(np.float32)
        a[y0:y1, x0:x1, :3] = np.clip(sa + (sb - sa) * m3, 0, 255).astype(np.uint8)
        return a

    def _process_dlss(self, live, rgba, bgr_src, reset, settings, mask=None, out_buf=None):
        """处理一帧。开启"皮肤双通道"时跑两遍：
        A 通道 skin_struct=0(基准)，B 通道按滑条目标值，
        用皮肤遮罩把 B 的效果限制在皮肤区域混合输出。
        bgr_src 用于计算遮罩(与 rgba 同尺寸)；settings 由调用者传入
        (工作线程里不能读 tkinter 变量)。mask 可预计算传入(流水线场景)。
        out_buf 为调用方拥有的输出缓冲: 需要跨帧持有结果(导出流水线)时必须传,
        否则结果位于 live 内部缓冲, 会被下一帧覆盖。"""
        s = settings
        enabled = bool(s.get('skin_dual')) and float(s.get('skin_struct', 0.0)) != 0.0
        if not enabled:
            o = live.process(rgba, reset=reset, out=out_buf)
            if o is not None and s.get('show_mask'):
                masker = self._ensure_masker()
                if masker is not None and masker.backend:
                    m = mask if mask is not None else self._mask_with_fallback(
                        masker, bgr_src, s.get('skin_params'))
                    o = cv2.cvtColor(masker.overlay(
                        cv2.cvtColor(o[..., :3], cv2.COLOR_RGB2BGR), m), cv2.COLOR_BGR2RGB)
                    o = np.dstack([o, np.full(o.shape[:2], 255, np.uint8)])
            return o
        masker = self._ensure_masker()
        if masker is None or not masker.backend:
            return live.process(rgba, reset=reset, out=out_buf)
        mask = mask if mask is not None else self._mask_with_fallback(masker, bgr_src, s.get('skin_params'))
        if float(mask.max()) < 0.05:
            # 未检出皮肤：单通道照常处理（skin_struct 在别处本来就无对象可作用）
            return live.process(rgba, reset=reset, out=out_buf)
        # 实测: 离线接入下 NGX 的 skin_struct 参数完全惰性(差异 0.000)，它被引擎皮肤遮罩门控；
        # 而 local_struct 是活的。所以 B 通道用"用户本地结构 + 皮肤滑条增量"驱动神经模型的
        # 结构增强，再用遮罩把增量限制在皮肤区域——等价于让"皮肤结构"只对皮肤生效。
        base_s = dict(s); base_s['skin_struct'] = 0.0
        live.update(base_s)
        a_buf = out_buf if out_buf is not None else self._dual_buf(rgba)
        a = live.process(rgba, reset=reset, out=a_buf)   # 通道A: 基准, 时域状态正常推进
        hi_s = dict(s)
        ls = float(s.get('local_struct', 1.0))
        hi_s['local_struct'] = max(-2.0, min(5.0, ls + float(s.get('skin_struct', 0.0))))
        live.update(hi_s)
        b = live.process(rgba, reset=False)          # 通道B: 同帧重跑(皮肤区内结构增强)
        live.update(dict(s))                         # 恢复用户设置，供后续单通道帧使用
        if a is None or b is None:
            got = b if a is None else a
            if got is None:
                return None
            if got is not a_buf:
                np.copyto(a_buf, got)                # 不能直接回传 live 内部缓冲
                got = a_buf
            return got
        self._maybe_check_dual_delta(a, b, mask)
        out = self._blend_mask(a, b, mask)
        if s.get('show_mask'):
            ov = masker.overlay(cv2.cvtColor(out[..., :3], cv2.COLOR_RGB2BGR), mask)
            out = np.dstack([cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), out[..., 3:4]])
        return out

    def _maybe_check_dual_delta(self, a, b, mask):
        """一次性自检：两通道在皮肤区的差异≈0 说明 skin_struct 在当前接入下完全惰性，
        双通道混合不可能产生效果——及时告知用户，避免白跑两遍。"""
        if self._dual_delta_checked:
            return
        self._dual_delta_checked = True
        try:
            sel = mask > 0.5
            if not np.any(sel):
                return
            d = float(np.abs(a[..., :3].astype(np.float32)[sel]
                             - b[..., :3].astype(np.float32)[sel]).mean())
            if d < 0.5:
                self._log_safe("[皮肤双通道] 警告: 两通道在皮肤区几乎无差异(%.2f)——"
                               "可能“本地结构”已到上限 5，皮肤增量无法再叠加。" % d)
            else:
                self._log_safe("[皮肤双通道] 已生效: 皮肤区两通道平均差异 %.1f/255" % d)
        except Exception:
            pass

    # ---------- realtime playback ----------
    @staticmethod
    def _playback_dims(w, h):
        if w <= 1280 and h <= 720:
            return w, h
        s = min(1280 / w, 720 / h)
        return max(2 * int(w * s / 2), 2), max(2 * int(h * s / 2), 2)

    def _pb_now_frame(self):
        if self._has_audio and self._audio is not None and self._audio.ok:
            if not self._audio.eof:
                return int(self._audio.time() * self.fps)
            if not self._pb_audio_reanchored:
                # audio ended before the video — switch to wall clock once
                self._pb_audio_reanchored = True
                self._pb_anchor_frame = int(self._audio.time() * self.fps)
                self._pb_anchor_wall = time.perf_counter()
                self._log_safe("[播放] 音频已结束，切换视频时钟")
        return int(self._pb_anchor_frame +
                   (time.perf_counter() - self._pb_anchor_wall) * self.fps * self._pb_speed)

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def on_space(self, event=None):
        # If a text field or a button has focus, let IT handle the space (type a space /
        # activate the button) — don't hijack it. Otherwise space toggles play/pause.
        w = self.root.focus_get()
        if w is not None:
            cls = w.winfo_class()
            if cls in ("Entry", "TEntry", "Text", "Combobox", "TCombobox", "Spinbox", "TSpinbox",
                       "Button", "TButton", "Checkbutton", "TCheckbutton", "Radiobutton", "TRadiobutton"):
                return None
        self.toggle_play()
        return "break"

    def _set_play_btn(self, playing):
        try:
            self.play_btn.config(text="⏸ 暂停" if playing else "▶ 播放")
        except Exception:
            pass

    def play(self):
        if self._sc_active:
            return
        if not self.video:
            messagebox.showwarning("提示", "请先导入视频/图片"); return
        if self.media_kind != "video":
            return
        if getattr(self, "_exporting", False):
            self.logln("[播放] 导出进行中，不能播放"); return
        try:
            speed = float(self.speed_var.get())
        except Exception:
            speed = 1.0
        speed = max(0.05, min(4.0, speed))
        self._pb_speed = speed
        w, h = self._media_w, self._media_h
        start = int(self.fslider.get())
        if start >= max(self.nframes - 1, 0):
            start = 0
        if speed < 1.0:
            pw, ph = max(2 * int(w / 2), 2), max(2 * int(h / 2), 2)
            self.logln(f"[播放] 慢速 {speed:.2f}x: 不降分辨率，按 {pw}x{ph} 处理")
        else:
            pw, ph = self._playback_dims(w, h)
            if (pw, ph) != (w, h):
                self.logln(f"[播放] 实时模式: 处理分辨率 {pw}x{ph} (源 {w}x{h})")
            else:
                self.logln("[播放] 实时模式: 源 ≤720p，按原尺寸处理")
        settings = self._collect_settings()
        self._pb_latest = None
        self._pb_draw_key = None
        self._pb_stop.clear()
        self._pb_seek_req = None
        self._pb_settings_pending = None
        self._pb_lag_logged = False
        self._pb_audio_reanchored = False
        if self._has_audio and self._audio is not None:
            self._audio.play(self.video, start / max(self.fps, 1.0), speed=speed)
            self._audio.set_volume(float(self.vol_var.get()))
            if self._audio.ok:
                self.logln(f"[音频] 从 {start / max(self.fps, 1.0):.2f}s 开始")
            else:
                self.logln("[音频] 启动失败，静音播放(视频时钟)")
        elif self._has_audio:
            self.logln("[音频] sounddevice 未安装，无声播放")
        self._pb_anchor_frame = start
        self._pb_anchor_wall = time.perf_counter()
        self.playing = True
        self._set_play_btn(True)
        self._pb_thread = threading.Thread(
            target=self._pb_worker,
            args=(self.video, start, pw, ph, settings), daemon=True)
        self._pb_thread.start()
        self._pb_tick()

    def _pb_worker(self, path, start_frame, pw, ph, settings):
        """Playback worker: owns the Live session for its lifetime. Sequential decode,
        downscale to pw x ph, DLSS-process, publish the latest frame pair."""
        try:
            live = self._ensure_live(pw, ph, settings)
        except Exception as ex:
            self._log_safe("[播放] DLSS 初始化失败: " + str(ex))
            live = None
        if live is None:
            self._log_safe("[播放] DLSS 引擎初始化失败，停止播放")
            self.playing = False
            self._pb_btn_pending = True
            return
        cap = cv2.VideoCapture(path)
        idx = start_frame
        if idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        need_reset = True
        cur_settings = settings
        # P0/P4: 播放应用当前档位(时域复用降遮罩成本)，从头保序起算遮罩时域状态
        masker = self._ensure_masker()
        if masker is not None:
            masker.reset()
            self._apply_perf_profile()
        try:
            while not self._pb_stop.is_set() and self.playing:
                if self._pb_seek_req is not None:
                    idx = int(self._pb_seek_req)
                    self._pb_seek_req = None
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    need_reset = True
                    if self._masker is not None:
                        self._masker.reset()
                s = self._pb_settings_pending
                if s is not None:
                    self._pb_settings_pending = None
                    try:
                        live.update(s)
                        cur_settings = s
                    except Exception as ex:
                        self._log_safe("[播放] 参数更新失败 " + str(ex))
                ok, fr = cap.read()
                if not ok:
                    time.sleep(0.02)      # EOF: wait for the loop-wrap seek request
                    continue
                if fr.shape[1] != pw or fr.shape[0] != ph:
                    fr_s = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA)
                else:
                    fr_s = fr
                rgb = cv2.cvtColor(fr_s, cv2.COLOR_BGR2RGB)
                rgba = np.dstack([rgb, np.full((ph, pw), 255, np.uint8)])
                o = self._process_dlss(live, rgba, fr_s, need_reset, cur_settings)
                need_reset = False
                if o is not None:
                    bgr = cv2.cvtColor(o[..., :3], cv2.COLOR_RGB2BGR)
                    with self._pb_lock:
                        self._pb_latest = (idx, fr_s, bgr)
                idx += 1
                try:
                    lead = idx - self._pb_now_frame()
                    if lead > 3:
                        time.sleep(min(lead * 0.005, 0.05))   # settle near real-time pace
                except Exception:
                    pass
        finally:
            cap.release()

    def _pb_tick(self):
        if not self.playing:
            return
        f = self._pb_now_frame()
        last = max(self.nframes - 1, 0)
        if f >= self.nframes:
            # loop wrap: restart worker read position + audio together
            self._pb_seek_req = 0
            if self._has_audio and self._audio is not None and self._audio.ok:
                self._audio.seek(0.0)
            self._pb_anchor_frame = 0
            self._pb_anchor_wall = time.perf_counter()
            self._pb_audio_reanchored = False
            f = 0
        f = max(0, min(f, last))
        self._pb_ui_suppress = True
        try:
            self.fslider.set(f)
            self.sync_frame_entry()
        except Exception:
            pass
        self._pb_ui_suppress = False
        lat = self._pb_latest
        if lat is not None:
            lag = f - lat[0]
            if lag > self.fps and not self._pb_lag_logged:
                self._pb_lag_logged = True
                self.logln("[播放] GPU 处理跟不上，保持最新帧显示（播放头按原速）")
            elif 0 <= lag <= 2 and self._pb_lag_logged:
                self._pb_lag_logged = False
                self.logln("[播放] 已追上")
        self._draw_playback_frame()
        self._pb_after = self.root.after(16, self._pb_tick)

    def _draw_playback_frame(self):
        with self._pb_lock:
            lat = self._pb_latest
        cw = max(self.canvas.winfo_width() or 780, 200)
        ch = max(self.canvas.winfo_height() or 400, 150)
        if lat is None:
            self.canvas.delete("all")
            self.canvas.create_text(cw // 2, ch // 2, text="处理中…", fill="#888888",
                                    font=("Microsoft YaHei", 11))
            self._pb_draw_key = None
            return
        idx, orig, dlss = lat
        view = self.view_var.get()
        key = (idx, view, round(self.split_x, 3), cw, ch)
        if key == self._pb_draw_key:
            return
        self._pb_draw_key = key
        if view == "原图":
            self._draw_fit(orig, cw, ch)
        elif view == "DLSS":
            self._draw_fit(dlss, cw, ch)
        else:
            self._compose_split(orig, dlss, cw, ch)

    def pause(self):
        was_playing = self.playing
        self.playing = False
        self._set_play_btn(False)
        if self._pb_after is not None:
            try:
                self.root.after_cancel(self._pb_after)
            except Exception:
                pass
            self._pb_after = None
        if self._pb_thread is not None:
            self._pb_stop.set()
            try:
                self._pb_thread.join(timeout=3.0)
            except Exception:
                pass
            self._pb_thread = None
        if self._audio is not None:
            self._audio.pause()
        try:
            cur = int(self.fslider.get())
        except Exception:
            cur = 0
        self._pb_anchor_frame = cur
        self._pb_anchor_wall = time.perf_counter()
        self._pb_latest = None
        self._pb_draw_key = None
        self._live_cache = None
        self._last_dlss_frame = -1
        self._split_frame = -1
        if was_playing and self.video and self.media_kind == "video":
            self.logln("[播放] 暂停，恢复全分辨率预览")
            try:
                self.display_view()
            except Exception as ex:
                self.logln(f"[preview] {ex}")

    # ---------- realtime screen-capture test ----------
    def _sc_toggle(self):
        if self._sc_active:
            self._sc_stop_capture()
        else:
            self._sc_start()

    def _sc_start(self):
        if not SCREEN_CAP_AVAILABLE or not self._sc_monitors:
            messagebox.showwarning("提示", "截屏模块不可用"); return
        if getattr(self, "_exporting", False):
            messagebox.showinfo("忙", "导出进行中，请稍候"); return
        if self._sc_region is None:
            self._sc_pick_region(on_done=self._sc_begin_capture)
            return
        self._sc_begin_capture()

    def _sc_begin_capture(self):
        idx = max(self.sc_mon_cb.current(), 0)
        ml, mt, mr, mb = self._sc_monitors[idx]["rect"]
        l, t, r, b = self._sc_region
        l, t = max(l, ml), max(t, mt)
        r, b = min(r, mr), min(b, mb)
        if r - l < 16 or b - t < 16:
            self._sc_region = None
            self.sc_region_label.config(text="未选区域")
            messagebox.showinfo("提示", "已选区域不在当前显示器内，请重新框选")
            self._sc_pick_region(on_done=self._sc_begin_capture)
            return
        region = (l, t, r, b)
        self.pause()
        try:
            cap = screen_cap.ScreenCapture(idx, region=region, log=self._log_safe)
        except Exception as ex:
            messagebox.showerror("截屏失败", str(ex)); return
        rw = cap.rect[2] - cap.rect[0]; rh = cap.rect[3] - cap.rect[1]
        pw, ph = self._playback_dims(rw, rh)
        self.logln(f"[截屏] 开始: {self._sc_monitors[idx]['device']} "
                   f"区域 ({l},{t})-({r},{b}) {rw}x{rh} "
                   f"后端={cap.backend}，处理分辨率 {pw}x{ph}")
        settings = self._collect_settings()
        self._sc_cap = cap
        with self._sc_lock:
            self._sc_latest = None
        self._sc_count = 0
        self._sc_draw_key = None
        self._sc_stop.clear()
        self._sc_settings_pending = None
        self._sc_fps = 0.0
        self._sc_active = True
        self._set_media_controls_enabled(False)
        self.sc_btn.config(text="⏹ 停止截屏测试")
        self._sc_thread = threading.Thread(
            target=self._sc_worker, args=(cap, pw, ph, settings), daemon=True)
        self._sc_thread.start()
        self._sc_tick()

    def _sc_pick_region(self, on_done=None):
        """全屏半透明覆盖层上拖拽框选截取区域；ESC/右键取消。"""
        if not SCREEN_CAP_AVAILABLE or not self._sc_monitors:
            messagebox.showwarning("提示", "截屏模块不可用"); return
        if self._sc_active:
            messagebox.showinfo("提示", "请先停止截屏测试"); return
        idx = max(self.sc_mon_cb.current(), 0)
        ml, mt, mr, mb = self._sc_monitors[idx]["rect"]
        self.pause()
        ov = tk.Toplevel(self.root)
        ov.attributes("-topmost", True)
        ov.overrideredirect(True)
        ov.geometry("%dx%d+%d+%d" % (mr - ml, mb - mt, ml, mt))
        ov.configure(bg="black")
        try:
            ov.attributes("-alpha", 0.35)
        except Exception:
            pass
        cv = tk.Canvas(ov, cursor="cross", bg="black", highlightthickness=0)
        cv.pack(fill="both", expand=True)
        cv.create_text((mr - ml) // 2, 40,
                       text="按住左键拖拽，框选要实时渲染的屏幕区域（ESC 或右键取消）",
                       fill="white", font=("Microsoft YaHei", 12, "bold"))
        st = {"x0": 0, "y0": 0, "rid": None}

        def finish(region):
            try:
                ov.destroy()
            except Exception:
                pass
            if region is None:
                return
            self._sc_region = region
            self.sc_region_label.config(text="已选 %dx%d @(%d,%d)" % (
                region[2] - region[0], region[3] - region[1],
                region[0], region[1]))
            self.logln("[截屏] 已选区域: (%d,%d)-(%d,%d)" % region)
            if on_done is not None:
                on_done()

        def press(e):
            st["x0"], st["y0"] = e.x, e.y
            if st["rid"] is not None:
                cv.delete(st["rid"])
            st["rid"] = cv.create_rectangle(e.x, e.y, e.x, e.y,
                                            outline="#ff3b30", width=2)

        def drag(e):
            if st["rid"] is not None:
                cv.coords(st["rid"], st["x0"], st["y0"], e.x, e.y)

        def release(e):
            x0, x1 = sorted((st["x0"], e.x)); y0, y1 = sorted((st["y0"], e.y))
            if x1 - x0 < 16 or y1 - y0 < 16:
                finish(None)
                return
            finish((ml + x0, mt + y0, ml + x1, mt + y1))

        def cancel(e=None):
            finish(None)

        cv.bind("<ButtonPress-1>", press)
        cv.bind("<B1-Motion>", drag)
        cv.bind("<ButtonRelease-1>", release)
        cv.bind("<ButtonPress-3>", cancel)
        ov.bind("<Escape>", cancel)
        ov.focus_force()

    def _sc_worker(self, cap, pw, ph, settings):
        """Capture worker: owns the Live session while active. Grab → downscale →
        DLSS-process → publish latest (orig_small, dlss_small)."""
        try:
            live = self._ensure_live(pw, ph, settings)
        except Exception as ex:
            self._log_safe("[截屏] DLSS 初始化失败: " + str(ex))
            live = None
        if live is None:
            self._log_safe("[截屏] DLSS 引擎初始化失败，停止")
            self._sc_stop_pending = True
            return
        need_reset = True
        n = 0
        cur_settings = settings
        t_win = time.perf_counter()
        try:
            while not self._sc_stop.is_set():
                s = self._sc_settings_pending
                if s is not None:
                    self._sc_settings_pending = None
                    try:
                        live.update(s)
                        cur_settings = s
                    except Exception as ex:
                        self._log_safe("[截屏] 参数更新失败 " + str(ex))
                fr = cap.grab()
                if fr is None:
                    time.sleep(0.01)
                    continue
                if fr.shape[1] != pw or fr.shape[0] != ph:
                    fr_s = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA)
                else:
                    fr_s = fr
                rgb = cv2.cvtColor(fr_s, cv2.COLOR_BGR2RGB)
                rgba = np.dstack([rgb, np.full((ph, pw), 255, np.uint8)])
                o = self._process_dlss(live, rgba, fr_s, need_reset, cur_settings)
                need_reset = False
                if o is not None:
                    bgr = cv2.cvtColor(o[..., :3], cv2.COLOR_RGB2BGR)
                    with self._sc_lock:
                        self._sc_latest = (fr_s, bgr)
                        self._sc_count += 1
                    n += 1
                    now = time.perf_counter()
                    if now - t_win >= 1.0:
                        self._sc_fps = n / (now - t_win)
                        n = 0
                        t_win = now
        except Exception as ex:
            self._log_safe("[截屏] 工作线程异常: " + str(ex))
        finally:
            try:
                cap.close()
            except Exception:
                pass

    def _sc_tick(self):
        if not self._sc_active:
            return
        try:
            self.sc_fps_label.config(text="渲染 %.1f fps" % self._sc_fps)
        except Exception:
            pass
        self._draw_sc_frame()
        self._sc_after = self.root.after(16, self._sc_tick)

    def _draw_sc_frame(self):
        with self._sc_lock:
            lat = self._sc_latest
            cnt = self._sc_count
        cw = max(self.canvas.winfo_width() or 780, 200)
        ch = max(self.canvas.winfo_height() or 400, 150)
        if lat is None:
            self.canvas.delete("all")
            self.canvas.create_text(cw // 2, ch // 2, text="截屏中…", fill="#888888",
                                    font=("Microsoft YaHei", 11))
            self._sc_draw_key = None
            return
        orig, dlss = lat
        view = self.view_var.get()
        key = (cnt, view, round(self.split_x, 3), cw, ch)
        if key == self._sc_draw_key:
            return
        self._sc_draw_key = key
        if view == "原图":
            self._draw_fit(orig, cw, ch)
        elif view == "DLSS":
            self._draw_fit(dlss, cw, ch)
        else:
            self._compose_split(orig, dlss, cw, ch)

    def _sc_stop_capture(self):
        if self._sc_thread is None and not self._sc_active:
            return
        self._sc_active = False
        if self._sc_after is not None:
            try:
                self.root.after_cancel(self._sc_after)
            except Exception:
                pass
            self._sc_after = None
        self._sc_stop.set()
        if self._sc_thread is not None:
            try:
                self._sc_thread.join(timeout=3.0)
            except Exception:
                pass
            self._sc_thread = None
        if self._sc_cap is not None:
            try:
                self._sc_cap.close()
            except Exception:
                pass
            self._sc_cap = None
        with self._sc_lock:
            self._sc_latest = None
        self._sc_draw_key = None
        self._live_cache = None
        self._last_dlss_frame = -1
        self._split_frame = -1
        self._set_media_controls_enabled(True)
        try:
            self.sc_btn.config(text="▶ 开始截屏测试")
            self.sc_fps_label.config(text="")
        except Exception:
            pass
        self.logln("[截屏] 已停止")
        try:
            self.display_view()
        except Exception:
            pass

    def _set_media_controls_enabled(self, enabled):
        st = "normal" if enabled else "disabled"
        for w in (self.fslider, self.fentry, self.play_btn, self.import_btn,
                  self.export_btn, self.crf_spin, self.sc_pick_btn):
            try:
                w.config(state=st)
            except Exception:
                pass
        for w in (self.sc_mon_cb, self.speed_cb):
            try:
                w.config(state="readonly" if enabled else "disabled")
            except Exception:
                pass

    # ---------- import ----------
    def import_media(self):
        p = filedialog.askopenfilename(filetypes=[
            ("视频/图片", "*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
            ("所有文件", "*.*")])
        if not p:
            return
        if getattr(self, "_exporting", False):
            messagebox.showinfo("忙", "导出进行中，请稍候"); return
        if self._sc_active:
            messagebox.showinfo("提示", "请先停止实时截屏测试"); return
        self.pause()
        # Do NOT close the Live here — NGX core init is one-time per process, so
        # closing + rebuilding it on the next preview crashes with an access violation.
        # Keep the shared Live alive; _ensure_live() will reuse it (same size) or
        # resize() it (new resolution) without re-running the init.
        self._live_cache = None
        self._last_dlss_frame = -1
        self._split_frame = -1
        if self._masker is not None:
            self._masker.reset()
        path = os.path.abspath(p)
        ext = os.path.splitext(path)[1].lower()
        if getattr(self, "_cap", None):
            self._cap.release()
        self._cap = None
        if self._audio is not None:
            self._audio.close()
        self._has_audio = False
        self._img = None
        if ext in IMAGE_EXTS:
            try:
                data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception as ex:
                img = None
                self.logln(f"[导入] 读取图片失败: {ex}")
            if img is None:
                messagebox.showerror("导入失败", "无法解码图片:\n" + path)
                return
            self.media_kind = "image"
            self.video = path
            self._img = img
            h, w = img.shape[:2]
            self._media_w, self._media_h = w, h
            self.nframes, self.fps = 1, 1.0
            self.vlabel.config(text=f"{os.path.basename(path)}  (图片 {w}x{h})")
            self.fslider.config(to=0, state="disabled")
            self.fslider.set(0)
            self.ftotal.config(text="/ 0")
            self.fentry.config(state="disabled")
            self.play_btn.config(state="disabled")
            self.crf_spin.config(state="disabled")
            self.export_btn.config(text="导出 DLSS 图片")
            self.logln(f"已导入图片: {path}  ({w}x{h})")
        else:
            self.media_kind = "video"
            self.video = path
            self._cap = cv2.VideoCapture(path)
            n, fps, w, h = self._video_info(path)
            self.nframes, self.fps = n, fps
            self._media_w, self._media_h = w, h
            self.vlabel.config(text=f"{os.path.basename(path)}  ({n} 帧 @ {fps:.0f}fps {w}x{h})")
            self.fslider.config(to=max(n - 1, 1), state="normal")
            self.fslider.set(0)
            self.ftotal.config(text=f"/ {max(n - 1, 1)}")
            self.fentry.config(state="normal")
            self.play_btn.config(state="normal")
            self.crf_spin.config(state="normal")
            self.export_btn.config(text="导出 DLSS 视频")
            self._has_audio = self._has_audio_stream(path)
            if self._has_audio:
                self.logln("[音频] 检测到音频流")
            elif self._audio is None:
                self.logln("[音频] sounddevice 未安装，无声播放")
            elif shutil.which("ffprobe") is None and shutil.which("ffmpeg") is None:
                self.logln("[音频] 未找到 ffmpeg/ffprobe，静音播放")
            else:
                self.logln("[音频] 该视频无音频流 — 播放用视频时钟")
            self.logln(f"已导入: {path}  ({n} 帧)")
        self.sync_frame_entry()
        try:
            self.display_view()
        except Exception as ex:
            self.logln(f"[preview] {ex}")
        self.set_status("就绪")

    @staticmethod
    def _video_info(path):
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return n, fps, w, h

    @staticmethod
    def _has_audio_stream(path):
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return False
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", path],
                capture_output=True, timeout=20, creationflags=_NO_WINDOW)
            return r.returncode == 0 and len(r.stdout.strip()) > 0
        except Exception:
            return False

    # ---------- export ----------
    @staticmethod
    def _compose_output(fr, o_bgr, view, mix=1.0):
        """Apply 输出视图 to one processed frame pair (BGR, same dims)."""
        ww = fr.shape[1]
        if view == 0 and mix >= 1.0:
            return o_bgr        # 默认视图: clip(ff+(fo-ff)*1) 恒等于 o_bgr, 省两次全画幅 float32
        fo = o_bgr.astype(np.float32); ff = fr.astype(np.float32)
        if view == 1:                          # 差异×10
            return (np.clip(0.5 + (fo - ff) / 255.0 * 10.0, 0, 1) * 255).astype(np.uint8)
        if view == 2:                          # 左右对比
            frame = o_bgr.copy()
            frame[:, :ww // 2] = fr[:, :ww // 2]
            if ww > 1:
                frame[:, max(ww // 2 - 1, 0)] = [255, 255, 255]
            return frame
        return np.clip(ff + (fo - ff) * mix, 0, 255).astype(np.uint8)

    def export_dlss(self):
        if not self.video:
            messagebox.showwarning("提示", "请先导入视频/图片"); return
        if self._sc_active:
            messagebox.showinfo("提示", "请先停止实时截屏测试"); return
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("忙", "上一个任务还没结束"); return
        if getattr(self, "_export_thread", None) is not None and self._export_thread.is_alive():
            messagebox.showinfo("忙", "上一个导出还没结束"); return
        self.pause()
        if self.media_kind == "image":
            self._export_image()
            return
        settings = self._collect_settings()
        n, fps, w, h = self._video_info(self.video)
        try:
            crf = int(self.crf.get())
        except Exception:
            crf = 18
        crf = max(0, min(51, crf))
        out_path = os.path.splitext(self.video)[0] + "_dlss.mp4"
        self._exporting = True
        self.set_status("正在导出...")
        # P1: 导出移出主线程(界面不再卡顿)，三级流水线: 解码∥遮罩∥GPU+编码
        self._export_thread = threading.Thread(
            target=self._export_worker, args=(settings, crf, out_path, n, fps), daemon=True)
        self._export_thread.start()

    def _export_worker(self, settings, crf, out_path, n, fps):
        """导出工作线程: 解码线程 + 遮罩线程池(P6 K 路交错) + GPU 线程四级流水线，
        解码/遮罩与 GPU 重叠执行；进度经主线程轮询回 UI。"""
        writer = None
        proc = None
        masker = self._ensure_masker()
        skin_dual = bool(settings.get('skin_dual')) and float(settings.get('skin_struct', 0.0)) != 0.0
        need_mask = masker is not None and masker.backend and (skin_dual or settings.get('show_mask'))
        params = settings.get('skin_params')
        view = settings['output_view']
        q_in = queue.Queue(maxsize=1)     # (i, rgba, fr, mask, reset) -> GPU 线程
        q_out = queue.Queue(maxsize=1)    # (i, o) 回传
        live_box = [None]
        gpu_err = [None]
        # 在飞帧环形缓冲。输入侧: 待送 GPU 的最多 K 帧(遮罩中) + q_in 1 + GPU 1,
        # 取 K+3 留余量; 输出侧只在 GPU 下游(GPU 在写 1 + q_out 1 + 主线程正写盘 1),
        # 与 K 无关永远是 3。若以后动队列深度或在飞深度, 两者必须同步加大,
        # 否则会把已排队帧的像素写花(无崩溃、只出错帧, 难发现)。
        RING_OUT = 3
        out_ring = [None] * RING_OUT

        def gpu_loop():
            try:
                while True:
                    item = q_in.get()
                    if item is None:
                        break
                    i, rgba, fr, mask, reset = item
                    if live_box[0] is None:
                        live_box[0] = self._ensure_live(rgba.shape[1], rgba.shape[0], settings)
                    if live_box[0] is None:
                        q_out.put(None)
                        continue
                    slot = i % RING_OUT
                    if out_ring[slot] is None or out_ring[slot].shape != rgba.shape:
                        out_ring[slot] = np.empty_like(rgba)
                    q_out.put((i, self._process_dlss(live_box[0], rgba, fr, reset, settings,
                                                     mask=mask, out_buf=out_ring[slot])))
            except Exception as ex:
                gpu_err[0] = ex
                q_out.put(None)

        wbuf = [None]

        def write_one(i, o, fr):
            nonlocal proc, writer
            if o is None:
                return
            if wbuf[0] is None or wbuf[0].shape[:2] != o.shape[:2]:
                wbuf[0] = np.empty((o.shape[0], o.shape[1], 3), np.uint8)
            cv2.cvtColor(o, cv2.COLOR_RGBA2BGR, dst=wbuf[0])   # 一次转换, dst 复用
            frame = self._compose_output(fr, wbuf[0], view)
            hh, ww = frame.shape[:2]
            if proc is None and writer is None:
                proc = self._start_ffmpeg(out_path, ww, hh, fps, crf)
                if proc is None:
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
            if proc is not None:
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                proc.stdin.write(memoryview(frame).cast("B"))  # 免一次全帧 tobytes 拷贝
            else:
                writer.write(frame)

        gpu_th = threading.Thread(target=gpu_loop, daemon=True)
        gpu_th.start()
        ex_dec = ThreadPoolExecutor(1)
        cap = cv2.VideoCapture(self.video)
        # ---- P6: 按档位/CPU/帧面积 定遮罩并行路数 K ----
        tier0 = getattr(self, "_gov_tier", None) or getattr(self, "_gpu_tier", "mid") or "mid"
        px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        maskers = self._ensure_maskers(perf_profile.mask_workers(tier0, px)) if need_mask else []
        if not maskers:
            maskers = [masker] if masker is not None else []
        K = max(1, len(maskers))
        # 实例只算无时域状态的每帧遮罩(profile 里 ema_alpha=0), EMA 权重拿出来在
        # 下面的有序消费点统一递推 -> 输出与单实例串行逐位一致。
        prof0, ema_a = perf_profile.parallel_profile(tier0, K)
        for mk in maskers:
            mk.apply_profile(prof0)
            mk.reset()
        ex_mask = ThreadPoolExecutor(K)
        RING_IN = K + 3
        rgba_ring = [None] * RING_IN
        prof_box = [prof0, 0]             # [待应用 profile, epoch]
        prof_seen = [0] * K               # 各实例已应用的 epoch
        ema_prev = [None]                 # 共享 EMA 的上一帧输出
        gov = perf_profile.Governor(getattr(self, "_gpu_tier", "mid") or "mid")
        applied_tier = gov.tier
        ft_ema, t_last, t_gov = 0.0, time.perf_counter(), 0.0
        i = 0
        pend_fr = None
        failed = False
        try:
            fut_dec = [ex_dec.submit(cap.read)]
            inflight = deque()
            nxt_idx = [0]
            eof = [False]

            def submit_prep(idx, fr_src):
                """组装 rgba(环形缓冲复用) + 把该帧遮罩提交给第 idx%K 个实例。

                每个实例同时最多 1 个任务: inflight 深度 <= K 保证队列里是连续 K 帧,
                idx%K 互不相同; 而提交 idx+K 之前 idx 的 future 已被 result() 过。
                各实例只算不含时域状态的每帧遮罩, 且 mask() 每次返回新数组,
                上一帧的遮罩不会被下一帧改写, 可以安全地交给 GPU 线程并行使用。
                """
                slot = idx % RING_IN
                if (rgba_ring[slot] is None
                        or rgba_ring[slot].shape[:2] != fr_src.shape[:2]):
                    rgba_ring[slot] = np.empty((fr_src.shape[0], fr_src.shape[1], 4), np.uint8)
                cv2.cvtColor(fr_src, cv2.COLOR_BGR2RGBA, dst=rgba_ring[slot])
                fut = None
                if need_mask:
                    w = idx % K
                    if prof_seen[w] != prof_box[1]:   # 治理器改档: 此刻该实例必然空闲
                        maskers[w].apply_profile(prof_box[0])
                        prof_seen[w] = prof_box[1]
                    fut = ex_mask.submit(maskers[w].mask, fr_src, params=params)
                return [idx, rgba_ring[slot], fr_src, fut]

            def fill():
                """把在飞帧补到 K 深: K 路遮罩实例只有喂满才并行得起来。"""
                while len(inflight) < K and not eof[0]:
                    ok, fr = fut_dec[0].result()
                    if not ok:
                        eof[0] = True
                        break
                    fut_dec[0] = ex_dec.submit(cap.read)
                    inflight.append(submit_prep(nxt_idx[0], fr))
                    nxt_idx[0] += 1

            fill()
            while inflight:
                idx, rgba, fr_cur, fut_mask = inflight.popleft()
                mask = fut_mask.result() if fut_mask is not None else None
                if mask is not None and ema_a > 0.0:
                    # 时域 EMA 在这里做(严格帧序、单一状态), 而不是在各实例内部:
                    # 每实例一条 EMA 链会让相邻帧来自不同链, 实测闪烁升到 1.4~1.7 倍。
                    p = ema_prev[0]
                    if p is not None and p.shape == mask.shape:
                        mask = ema_a * p + (1.0 - ema_a) * mask
                    ema_prev[0] = mask
                q_in.put((idx, rgba, fr_cur, mask, idx == 0))
                # 先把遮罩流水线补满, 再去等 GPU 结果: 否则 K 路实例喂不满,
                # 且遮罩_{i+1} 无法与 GPU_i 重叠(退化成同步调用)。
                fill()
                if pend_fr is not None:
                    res = q_out.get()
                    if res is None:
                        failed = True
                        break
                    write_one(res[0], res[1], pend_fr)
                    if res[0] % 5 == 0:
                        self._export_prog = (res[0], n)
                pend_fr = fr_cur
                i = idx + 1
                # P4 动态治理: 每 2s 采一次 GPU 利用率, 滞后升降档(帧耗时<3帧时长算达标)
                now = time.perf_counter()
                ft = now - t_last
                t_last = now
                ft_ema = ft if ft_ema <= 0 else 0.9 * ft_ema + 0.1 * ft
                if now - t_gov > 2.0:
                    t_gov = now
                    gov.sample(perf_profile.gpu_util(), ft_ema < 3.0 / max(fps, 1e-3))
                    if gov.tier != applied_tier:
                        self._gov_tier = gov.tier
                        # 遮罩是 K 路并行的: 不能在主线程直接改实例属性(会与正在跑的
                        # mask() 竞争), 也不能靠往线程池提交任务(不保证落到目标实例)。
                        # 只改 epoch, 由 submit_prep 在该实例确定空闲时补应用。
                        prof_box[0], ema_a = perf_profile.parallel_profile(gov.tier, K)
                        prof_box[1] += 1
                        applied_tier = gov.tier
                        self._log_safe("[性能] 自动调档 -> %s (利用率自适应)" % perf_profile.TIER_NAME[applied_tier])
            q_in.put(None)
            if pend_fr is not None and not failed:
                res = q_out.get()
                if res is not None:
                    write_one(res[0], res[1], pend_fr)
            gpu_th.join(timeout=15)
            if failed or gpu_err[0] is not None:
                self._log_safe("[导出] GPU 处理失败: " + str(gpu_err[0] or "无输出"))
        except (BrokenPipeError, OSError) as pe:
            self._log_safe("[导出] ffmpeg 写入中断: " + str(pe))
        except Exception as ex:
            self._log_safe("导出错误: " + str(ex))
        finally:
            ex_dec.shutdown(wait=False)
            ex_mask.shutdown(wait=False)
            cap.release()
            # 还原预览语义: 导出期给实例的 profile 把 ema_alpha 置了 0(EMA 外提),
            # 预览仍靠实例内部的 EMA, 必须把档位 profile 重新应用回去。
            self._park_extra_maskers()
            if masker is not None:
                masker.reset()
            self._apply_perf_profile()
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    rc = proc.wait()
                    err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
                    if rc != 0:
                        self._log_safe("[导出] ffmpeg 退出码 %s:\n%s" % (rc, err[-600:]))
                except Exception:
                    pass
            try:
                if writer:
                    writer.release()
            except Exception:
                pass
            done = (i > 0 and not failed)
            self._export_done = (out_path, done)

    def _export_finish(self, out_path, ok):
        self._exporting = False
        try:
            self.pbar["value"] = 0
        except Exception:
            pass
        if ok:
            self.set_status("完成")
            self.logln("已导出: " + out_path)
            messagebox.showinfo("导出", "已导出:\n" + out_path)
        else:
            self.set_status("导出失败")
            messagebox.showwarning("导出", "导出失败，详见日志")

    def _export_image(self):
        settings = self._collect_settings()
        fr = self._img
        if fr is None:
            return
        h, w = fr.shape[:2]
        self.set_status("正在处理图片…")
        self._exporting = True
        try:
            live = self._ensure_live(w, h, settings)
            if live is None:
                self.logln("DLSS 引擎初始化失败")
                return
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
            o = self._process_dlss(live, rgba, fr, True, settings)
            if o is None:
                self.logln("DLSS 处理失败")
                return
            o_bgr = cv2.cvtColor(o[..., :3], cv2.COLOR_RGB2BGR)
            frame = self._compose_output(fr, o_bgr, settings['output_view'])
            out_path = os.path.splitext(self.video)[0] + "_dlss.png"
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                self.logln("PNG 编码失败")
                return
            buf.tofile(out_path)
            self.set_status("完成")
            self.logln("已导出: " + out_path)
            messagebox.showinfo("导出", "已导出:\n" + out_path)
        except Exception as ex:
            traceback.print_exc()
            self.logln("导出错误: " + str(ex))
        finally:
            self._exporting = False

    def _start_ffmpeg(self, out_path, w, h, fps, crf):
        """Forward processed BGR frames to ffmpeg (libx264 + CRF) for real mp4 quality control.
        Returns a Popen with a stdin pipe, or None if ffmpeg is unavailable / dims are odd
        (yuv420p needs even dimensions) — the caller then falls back to OpenCV mp4v."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._log_safe("[导出] 未找到 ffmpeg，回退 OpenCV mp4v（“导出质量”将不生效）")
            return None
        if (w % 2) or (h % 2):
            self._log_safe("[导出] 视频尺寸 %dx%d 非偶数，ffmpeg 需要偶数，回退 OpenCV mp4v" % (w, h))
            return None
        crf = int(max(0, min(51, crf)))
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % (w, h), "-r", "%.6f" % fps,
               "-i", "-",
               "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
               "-pix_fmt", "yuv420p",
               "-movflags", "+faststart",
               out_path]
        self._log_safe("[导出] ffmpeg 编码: CRF=%s -> %s" % (crf, out_path))
        try:
            return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=_NO_WINDOW)
        except Exception as ex:
            self._log_safe("[导出] 启动 ffmpeg 失败: " + str(ex))
            return None

    def _iter_frames(self):
        cap = cv2.VideoCapture(self.video)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield i, f
            i += 1
        cap.release()

    # ---------- close ----------
    def on_close(self):
        try:
            self.pause()
        except Exception:
            pass
        try:
            self._sc_stop_capture()
        except Exception:
            pass
        if self._audio is not None:
            try:
                self._audio.close()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    if "--selftest" in sys.argv:
        # headless DLSS sanity check (writes a result file; used to verify the frozen exe
        # can load the host/runtime DLLs from _MEIPASS and actually run Feature 18).
        import dlss_engine
        try:
            f = np.full((360, 640, 3), 90, np.uint8)
            cv2.circle(f, (320, 180), 80, (255, 0, 0), -1)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, np.full((360, 640), 255, np.uint8)])
            live = dlss_engine.Live(640, 360, {'style': 1, 'intensity': 0.9})
            o = live.process(rgba, reset=True)
            live.close()
            ok = "DLSS_OK " + (str(o.shape) if o is not None else "None")
        except Exception as e:
            ok = "DLSS_FAIL " + repr(e)[:300]
        # 皮肤遮罩链自检: mediapipe 后端在冻结包里失败会静默回退 OpenCV(覆盖率极低,
        # 自动调优返回默认值)，加一段显式检测，打包后跑 --selftest 即可远程定位。
        try:
            import skin_mask
            mk = skin_mask.SkinMasker()
            f2 = np.full((360, 640, 3), 60, np.uint8)
            cv2.circle(f2, (320, 180), 90, (140, 160, 225), -1)   # 肤色圆块(BGR)
            m = mk.mask(f2)
            cov = float((m > 0.5).mean()) * 100.0
            if mk.backend == "mediapipe" and cov > 3.0:
                ok += "\nSKIN_OK backend=%s cov=%.1f%%" % (mk.backend, cov)
            else:
                ok += "\nSKIN_FAIL backend=%s cov=%.1f%% (期望 mediapipe 且 >3%%)" % (mk.backend, cov)
        except Exception as e:
            ok += "\nSKIN_FAIL " + repr(e)[:300]
        try:
            outdir = os.path.dirname(os.path.abspath(sys.argv[0]))
            with open(os.path.join(outdir, "_selftest.txt"), "w", encoding="utf-8") as fh:
                fh.write(ok)
        except Exception:
            pass
        return
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
