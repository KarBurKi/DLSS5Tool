#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dlss_engine.py — ctypes wrapper for the test4 DLSS5 Feature 18 host DLL (zero-guidance).

The Feature 18 ("neural render") ignores depth/flow guidance in this config, so this
engine always feeds ZERO guidance. It only needs the colour frames from the video.
"""
import ctypes
import os
import sys
import cv2
import numpy as np

# resolve the bundle dir: PyInstaller (frozen) -> _MEIPASS, else the script's own dir
if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS
    # NGX 日志落到 exe 旁边: _MEIPASS 是临时目录, 退出即删, 用户机上排障看不到
    LOG_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = BASE
HOST_DLL = os.path.join(BASE, "dlssnr_host.dll")
DLSSNR_DLL = os.path.join(BASE, "nvngx_dlssnr.dll")
LOG_PATH = os.path.join(LOG_DIR, "dlss_run.log")

_lib = None
_plugin_locked = False     # NGX 每进程只能 init 一次: 首次 dlssnr_init 后插件路径锁死


def set_plugin_dll(path):
    """在首次 dlssnr_init 前切换渲染插件 DLL(按 GPU 系列, 见 perf_profile.nvngx_dll_for)。
    返回是否生效; NGX 一旦初始化过(或路径不存在)再调用一律无效。"""
    global DLSSNR_DLL
    if _plugin_locked or not path or not os.path.exists(path):
        return False
    DLSSNR_DLL = path
    return True


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(HOST_DLL):
            raise FileNotFoundError("missing %s" % HOST_DLL)
        _lib = ctypes.CDLL(HOST_DLL)
        _lib.dlssnr_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p, ctypes.c_wchar_p]
        _lib.dlssnr_init.restype = ctypes.c_int
        _lib.dlssnr_create_feature.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.dlssnr_create_feature.restype = ctypes.c_int
        _lib.dlssnr_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        _lib.dlssnr_process.restype = ctypes.c_int
        _lib.dlssnr_set_options.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float]
        _lib.dlssnr_set_options.restype = None
        _lib.dlssnr_shutdown.argtypes = []
        _lib.dlssnr_resize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.dlssnr_resize.restype = ctypes.c_int
    return _lib


# settings dict keys that actually change the output on this Feature 18 config:
#   style (int), intensity (0..10), local_tone (0..10), local_struct (0..10),
#   skin_struct (0..10), output_view (0/1/2), output_mix (0..2).
#   preset/guidance/depth/flow are inert.
def _set_options(lib, s):
    lib.dlssnr_set_options(
        int(s.get('preset', 1)),          # preset: inert at same-res, keep 1
        int(s.get('style', 0)),
        float(s.get('intensity', 1.0)),
        float(s.get('local_tone', 1.0)),
        float(s.get('local_struct', 1.0)),
        float(s.get('skin_struct', 1.0)),
        int(s.get('use_auto_mask', 0)),    # inert
        int(s.get('ui_correction', 0)),    # inert
        0,                                 # guidance_mode ALWAYS 0 (off) — NR ignores guidance
        int(s.get('depth_convention', 2)), # inert (depth ignored)
        float(s.get('motion_scale_x', 1.0)),
        float(s.get('motion_scale_y', 1.0)))


def _apply_output_view(processed, color, view, mix, w, h):
    """Post-process the DLSS RGBA8 output per Output View (0=Processed,1=DiffX10,2=L/R Compare)."""
    out = []
    for pr, co in zip(processed, color):
        cof = co[..., :3].astype(np.float32) / 255.0
        prf = pr[..., :3].astype(np.float32) / 255.0
        if view == 1:      # Difference x10
            r = np.clip(0.5 + (prf - cof) * 10.0, 0, 1)
        elif view == 2:    # Left / Right compare
            r = prf.copy()
            r[:, :w // 2] = cof[:, :w // 2]
            if w % 2 == 1:
                r[:, w // 2] = 1.0
        else:              # Processed, blended by mix
            r = cof + (prf - cof) * mix
        res = np.dstack([r, np.ones((h, w), np.float32)])
        out.append((res * 255.0).clip(0, 255).astype(np.uint8))
    return out


def _read_log_tail(n=800):
    """读 dlss_run.log 尾部。host 持有日志句柄时读取会 PermissionError(实测),
    不能把真正的 NGX 错误吞掉。"""
    try:
        return open(LOG_PATH, errors="replace").read()[-n:] if os.path.exists(LOG_PATH) else ""
    except OSError:
        return "(dlss_run.log 被占用, 读不到)"


class Live:
    """Persistent single-frame DLSS session for realtime preview. init+create the feature
    once, then process() per frame. Style/intensity/local_* apply at the next process;
    changing 'preset' recreates the feature. close() releases the D3D12 device."""
    def __init__(self, w, h, settings=None):
        self._w, self._h = w, h
        self.settings = dict(settings or {})
        self._lib = _load()
        self._buf_shape = None      # P2: 复用零引导缓冲, 避免每帧分配(形状变了才重分)
        self._open()

    def _open(self):
        global _plugin_locked
        s = self.settings
        _set_options(self._lib, s)          # push preset before create
        try:
            self._lib.dlssnr_shutdown()
        except Exception:
            pass
        _plugin_locked = True               # 从此刻起插件 DLL 不能再换
        if not self._lib.dlssnr_init(self._w, self._h, int(s.get('preset', 1)), DLSSNR_DLL, LOG_PATH):
            raise RuntimeError("dlssnr_init failed (D3D12/gate). See dlss_run.log")
        if not self._lib.dlssnr_create_feature(self._w, self._h, int(s.get('preset', 1))):
            raise RuntimeError("Feature 18 create failed.\n" + _read_log_tail())

    def update(self, settings):
        old_preset = self.settings.get('preset')
        self.settings.update(settings)
        if self.settings.get('preset') != old_preset:
            self._open()

    def resize(self, w, h, preset=None):
        """Re-create the Feature 18 for a new frame size WITHOUT re-running the NGX core
        init (which is one-time per process and crashes if re-initialized)."""
        if preset is None:
            preset = int(self.settings.get('preset', 1))
        self.settings['preset'] = preset
        _set_options(self._lib, self.settings)
        if not self._lib.dlssnr_resize(w, h, preset):
            raise RuntimeError("Feature 18 resize failed.\n" + _read_log_tail())
        self._w, self._h = w, h

    def process(self, rgba, reset=False, out=None):
        """渲染一帧。out 为 None 时写入内部复用缓冲并返回它——注意该缓冲会被
        下一次 process 覆盖，调用方若需要跨帧持有结果(双通道 A/B、导出流水线)
        必须自带 out 缓冲，否则会读到后一帧的像素。"""
        _set_options(self._lib, self.settings)
        h, w = rgba.shape[:2]
        if self._buf_shape != (h, w, rgba.dtype):
            self._buf_shape = (h, w, rgba.dtype)
            self._mv = np.zeros((h, w, 2), np.float32)
            self._dp = np.zeros((h, w), np.float32)
            self._o = np.zeros_like(rgba)
        o = self._o
        if out is not None and out.shape == rgba.shape and out.dtype == rgba.dtype:
            o = out
        ok = self._lib.dlssnr_process(
            rgba.ctypes.data_as(ctypes.c_void_p),
            self._mv.ctypes.data_as(ctypes.c_void_p),
            self._dp.ctypes.data_as(ctypes.c_void_p),
            o.ctypes.data_as(ctypes.c_void_p),
            1 if reset else 0)
        return o if ok else None

    def close(self):
        try:
            self._lib.dlssnr_shutdown()
        except Exception:
            pass


class CpuLive:
    """纯 CPU 回退渲染(无 NVIDIA GPU / D3D12 不可用)。接口对齐 Live，
    用 OpenCV unsharp + 饱和度近似神经渲染增强，质量上限低于 Feature 18，
    仅供无 GPU 环境出图。皮肤双通道仍由调用方用遮罩混合实现。"""
    engine = "cpu"

    def __init__(self, w, h, settings=None):
        self._w, self._h = w, h
        self.settings = dict(settings or {})

    def update(self, settings):
        self.settings.update(settings)

    def resize(self, w, h, preset=None):
        self._w, self._h = w, h

    def process(self, rgba, reset=False, out=None):
        s = self.settings
        inten = float(s.get('intensity', 1.0)) / 5.0
        tone = float(s.get('local_tone', 1.0)) / 5.0
        bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
        blur = cv2.GaussianBlur(bgr, (0, 0), 3)
        out_bgr = cv2.addWeighted(bgr, 1.0 + 0.55 * inten, blur, -0.55 * inten, 0)
        hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + 0.12 * tone), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + 0.04 * tone), 0, 255)
        out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if out is not None and out.shape == rgba.shape and out.dtype == rgba.dtype:
            # dst 必须是连续整块, 故用 BGR2RGBA 一次写满再补回原 alpha
            cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGBA, dst=out)
            out[..., 3] = rgba[..., 3]
            return out
        return np.dstack([cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), rgba[..., 3]])

    def close(self):
        pass


def run_dlss(rgba_frames, settings=None, reset=True, progress=None):
    """Batch-generate DLSS for a list of HxWx4 rgba frames (zero guidance). Returns HxWx4 list."""
    settings = settings or {}
    lib = _load()
    global _plugin_locked
    _plugin_locked = True
    h, w = rgba_frames[0].shape[:2]
    _set_options(lib, settings)
    if not lib.dlssnr_init(w, h, int(settings.get('preset', 1)), DLSSNR_DLL, LOG_PATH):
        raise RuntimeError("dlssnr_init failed. See dlss_run.log")
    if not lib.dlssnr_create_feature(w, h, int(settings.get('preset', 1))):
        raise RuntimeError("Feature 18 create failed.\n" + _read_log_tail())
    out = []
    for i, rgba in enumerate(rgba_frames):
        mv = np.zeros((h, w, 2), np.float32)
        dp = np.zeros((h, w), np.float32)
        o = np.zeros_like(rgba)
        lib.dlssnr_process(
            rgba.ctypes.data_as(ctypes.c_void_p),
            mv.ctypes.data_as(ctypes.c_void_p),
            dp.ctypes.data_as(ctypes.c_void_p),
            o.ctypes.data_as(ctypes.c_void_p),
            1 if (reset and i == 0) else 0)
        if progress:
            progress(i, len(rgba_frames), "ok")
        out.append(o)
    lib.dlssnr_shutdown()
    view = settings.get('output_view', 0)
    mix = float(settings.get('output_mix', 1.0))
    if view != 0 or mix < 1.0:
        out = _apply_output_view(out, rgba_frames, view, mix, w, h)
    return out
