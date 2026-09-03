#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_out.py — ffmpeg 流式解码 + sounddevice 输出，为 gui.py 的播放提供可暂停/
可跳转的同步音频。

时钟约定：time() 返回"视频内绝对秒数"（已按播放速度换算）。输出流启动后由
回调逐块累计已送出的采样数；首个真实数据块到达前 time() 保持在起始位置（让
视频播放头等音频，避免启动延迟造成音画错位）。变速播放时用 ffmpeg atempo
同步拉伸/压缩音频，时钟仍按视频时间返回，音画不漂移。
"""
import collections
import shutil
import subprocess
import threading
import time

import numpy as np

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _atempo_chain(speed):
    """atempo only accepts 0.5-2.0 per instance; chain multiple for wider range."""
    parts = []
    s = float(speed)
    while s < 0.5:
        parts.append("atempo=0.5"); s /= 0.5
    while s > 2.0:
        parts.append("atempo=2.0"); s /= 2.0
    parts.append("atempo=%.4f" % s)
    return ",".join(parts)


class AudioPlayer:
    SR = 48000      # 输出采样率
    CH = 2          # 声道数
    CHUNK = 2400    # 每块样本数/声道 (50ms)

    def __init__(self, log=None):
        self._log = log or (lambda m: None)
        self._ffmpeg = shutil.which("ffmpeg")
        self._ok = SD_AVAILABLE and self._ffmpeg is not None
        self._proc = None
        self._feeder = None
        self._stream = None
        self._buf = collections.deque(maxlen=64)  # ~3.2s 上限
        self._partial = None
        self._base_t = 0.0
        self._samples_out = 0
        self._started = False
        self._eof = False
        self._playing = False
        self._run_ev = threading.Event()   # set=运行, clear=暂停(供数线程停读)
        self._lock = threading.Lock()
        self._volume = 1.0                 # 0.0 ~ 2.0 增益
        self._speed = 1.0                  # 播放速度（时钟换算用）

    # ---------- properties ----------
    @property
    def ok(self):
        return self._ok

    @property
    def eof(self):
        return self._eof

    @property
    def playing(self):
        return self._playing

    def set_volume(self, v):
        try:
            self._volume = max(0.0, min(2.0, float(v)))
        except Exception:
            pass

    def time(self):
        # 输出采样按 atempo 拉伸，采样数*速度 = 视频内时间
        return self._base_t + self._samples_out / float(self.SR) * self._speed

    # ---------- control ----------
    def play(self, path, t_seconds, speed=1.0):
        """从视频的 t_seconds 处开始播音频（重启 ffmpeg + 输出流）。
        speed != 1.0 时用 atempo 变速，时钟仍按视频时间返回。"""
        if not self._ok:
            return
        try:
            self._speed = max(0.05, min(4.0, float(speed)))
        except Exception:
            self._speed = 1.0
        with self._lock:
            self._kill_proc()
            self._stop_stream()
            self._buf.clear()
            self._partial = None
            self._eof = False
            self._started = False
            self._base_t = float(max(t_seconds, 0.0))
            self._samples_out = 0
            if not self._spawn(path, self._base_t):
                self._playing = False
                return
            self._run_ev.set()
            self._playing = True
            try:
                self._stream = sd.OutputStream(
                    samplerate=self.SR, channels=self.CH, dtype="int16",
                    blocksize=1024, latency="low", callback=self._callback)
                self._stream.start()
            except Exception as ex:
                self._ok = False
                self._playing = False
                self._kill_proc()
                self._log("[音频] 声卡输出启动失败(无声播放): %s" % ex)

    def pause(self):
        if not self._playing:
            return
        self._run_ev.clear()
        try:
            if self._stream is not None and self._stream.active:
                self._stream.stop()
        except Exception:
            pass

    def resume(self):
        if not self._playing:
            return
        self._run_ev.set()
        try:
            if self._stream is not None and not self._stream.active:
                self._stream.start()
        except Exception:
            pass

    def seek(self, t_seconds):
        """播放中跳转：杀掉 ffmpeg 在新位置重启（输出流保持，短暂静音）。"""
        if not self._ok or not self._playing:
            return
        with self._lock:
            self._kill_proc()
            self._buf.clear()
            self._partial = None
            self._eof = False
            self._started = False
            self._base_t = float(max(t_seconds, 0.0))
            self._samples_out = 0
            self._spawn(self._path, self._base_t)

    def close(self):
        """彻底关闭（导入新视频/退出时调用）。之后仍可再 play()。"""
        with self._lock:
            self._kill_proc()
        self._stop_stream()
        self._playing = False

    # ---------- internals ----------
    def _spawn(self, path, t):
        self._path = path
        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "error",
               "-ss", "%.6f" % t, "-i", path,
               "-vn", "-ac", str(self.CH), "-ar", str(self.SR)]
        if abs(self._speed - 1.0) > 1e-3:
            cmd += ["-af", _atempo_chain(self._speed)]
        cmd += ["-f", "s16le", "pipe:1"]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, creationflags=_NO_WINDOW)
        except Exception as ex:
            self._log("[音频] 启动 ffmpeg 失败: %s" % ex)
            self._proc = None
            return False
        self._feeder = threading.Thread(target=self._feed, daemon=True)
        self._feeder.start()
        return True

    def _kill_proc(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass

    def _stop_stream(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _feed(self):
        proc = self._proc
        nbytes = self.CHUNK * self.CH * 2
        cap = self._buf.maxlen
        while proc is not None and proc is self._proc:
            if not self._run_ev.is_set():
                time.sleep(0.05)
                continue
            if len(self._buf) >= cap - 2:
                time.sleep(0.02)      # buffer full: back-pressure ffmpeg instead of dropping
                continue
            try:
                data = proc.stdout.read(nbytes)
            except Exception:
                break
            if proc is not self._proc:
                break
            if not data:
                self._eof = True
                break
            arr = np.frombuffer(data, dtype=np.int16)
            arr = arr[:(arr.shape[0] // self.CH) * self.CH]
            if arr.size:
                self._buf.append(arr)

    def _callback(self, outdata, frames, time_info, status):
        filled = 0
        while filled < frames:
            if self._partial is None:
                if not self._buf:
                    break
                self._partial = self._buf.popleft()
            need = (frames - filled) * self.CH
            take = min(need, self._partial.shape[0])
            n = take // self.CH
            outdata[filled:filled + n] = self._partial[:take].reshape(n, self.CH)
            filled += n
            self._partial = self._partial[take:] if take < self._partial.shape[0] else None
        if filled < frames:
            outdata[filled:] = 0
        vol = self._volume
        if vol != 1.0:
            outdata[:] = np.clip(outdata.astype(np.float32) * vol,
                                 -32768, 32767).astype(np.int16)
        if filled > 0 or self._started:
            # 首个真实数据块之后时钟恒速推进（欠载时以静音补齐，不拖慢播放头）
            self._started = True
            self._samples_out += frames
