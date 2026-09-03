#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
screen_cap.py — 屏幕截取（供"实时截屏测试"使用）。

后端策略：优先 dxcam (DXGI Desktop Duplication，GPU 侧、快)；失败自动回退
纯 ctypes GDI BitBlt（任意 Windows 可用，1080p 约 5-15ms/帧）。
grab() 统一返回 BGR numpy 数组（OpenCV 格式），无新帧时返回 None。
"""
import ctypes
import ctypes.wintypes as wt

import numpy as np

try:
    import dxcam
    DXCAM_AVAILABLE = True
except Exception:
    DXCAM_AVAILABLE = False

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD),
                ("rcMonitor", wt.RECT),
                ("rcWork", wt.RECT),
                ("dwFlags", wt.DWORD),
                ("szDevice", wt.WCHAR * 32)]


_MONITORENUMPROC = ctypes.WINFUNCTYPE(
    wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM)


def list_monitors():
    """返回 [{'device': '\\\\.\\DISPLAY1', 'rect': (l, t, r, b)}, ...]"""
    mons = []

    def cb(hmon, hdc, rect_ptr, lparam):
        mi = _MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            r = mi.rcMonitor
            mons.append({"device": mi.szDevice,
                         "rect": (r.left, r.top, r.right, r.bottom)})
        return True

    user32.EnumDisplayMonitors(None, None, _MONITORENUMPROC(cb), 0)
    return mons


class _GdiCapture:
    """GDI BitBlt 截屏（缓存 DC/位图，重复抓取时只跑 BitBlt + GetDIBits）。"""

    def __init__(self, rect):
        self._rect = rect
        self._w = rect[2] - rect[0]
        self._h = rect[3] - rect[1]
        self._hdc = None
        self._mdc = None
        self._bmp = None
        self._info = None
        self._buf = None
        self._open()

    def _open(self):
        self._x, self._y = self._rect[0], self._rect[1]
        self._hdc = user32.GetDC(0)
        if not self._hdc:
            raise RuntimeError("GetDC failed")
        self._mdc = gdi32.CreateCompatibleDC(self._hdc)
        self._bmp = gdi32.CreateCompatibleBitmap(self._hdc, self._w, self._h)
        gdi32.SelectObject(self._mdc, self._bmp)

        class _BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wt.DWORD), ("biWidth", ctypes.c_long),
                        ("biHeight", ctypes.c_long), ("biPlanes", wt.WORD),
                        ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                        ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                        ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wt.DWORD),
                        ("biClrImportant", wt.DWORD)]

        bmi = _BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.biWidth = self._w
        bmi.biHeight = -self._h          # 负值 = 自上而下
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = BI_RGB
        self._info = bmi
        self._buf = ctypes.create_string_buffer(self._w * self._h * 4)

    def grab(self):
        if not gdi32.BitBlt(self._mdc, 0, 0, self._w, self._h,
                            self._hdc, self._x, self._y, SRCCOPY):
            return None
        n = gdi32.GetDIBits(self._mdc, self._bmp, 0, self._h,
                            self._buf, ctypes.byref(self._info), DIB_RGB_COLORS)
        if n == 0:
            return None
        arr = np.frombuffer(self._buf, np.uint8).reshape(self._h, self._w, 4)
        return arr[:, :, :3].copy()          # BGRA -> BGR

    def close(self):
        if self._bmp:
            gdi32.DeleteObject(self._bmp); self._bmp = None
        if self._mdc:
            gdi32.DeleteDC(self._mdc); self._mdc = None
        if self._hdc:
            user32.ReleaseDC(0, self._hdc); self._hdc = None


class ScreenCapture:
    """按显示器索引抓取整屏，或传入 region=(l, t, r, b) 只抓取框选区域。
    grab() -> BGR ndarray | None。close() 幂等。"""

    def __init__(self, monitor_index=0, region=None, log=None):
        self._log = log or (lambda m: None)
        mons = list_monitors()
        if not mons:
            raise RuntimeError("未检测到显示器")
        monitor_index = max(0, min(monitor_index, len(mons) - 1))
        ml, mt, mr, mb = mons[monitor_index]["rect"]
        if region is None:
            self.rect = (ml, mt, mr, mb)
        else:
            l = max(int(region[0]), ml); t = max(int(region[1]), mt)
            r = min(int(region[2]), mr); b = min(int(region[3]), mb)
            if r - l < 2 or b - t < 2:
                raise RuntimeError("截取区域过小或不在该显示器范围内")
            self.rect = (l, t, r, b)
        self.backend = None
        self._cam = None
        self._gdi = None
        if DXCAM_AVAILABLE:
            try:
                self._cam = dxcam.create(device_idx=0, output_idx=monitor_index,
                                         region=self.rect, output_color="BGR")
                self.backend = "dxcam"
                return
            except Exception as ex:
                self._log("[截屏] DXGI 后端不可用(%s)，回退 GDI" % type(ex).__name__)
                try:
                    if self._cam is not None:
                        self._cam.release()
                except Exception:
                    pass
                self._cam = None
        self._gdi = _GdiCapture(self.rect)
        self.backend = "gdi"

    def grab(self):
        if self._cam is not None:
            try:
                return self._cam.grab()
            except Exception as ex:
                self._log("[截屏] 抓取失败: %s" % ex)
                return None
        if self._gdi is not None:
            return self._gdi.grab()
        return None

    def close(self):
        try:
            if self._cam is not None:
                self._cam.release()
        except Exception:
            pass
        self._cam = None
        try:
            if self._gdi is not None:
                self._gdi.close()
        except Exception:
            pass
        self._gdi = None
