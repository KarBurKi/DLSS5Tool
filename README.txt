DLSS5 简约工具
================
导入视频/图片 → 实时预览(原图/DLSS/对比) → 调风格/强度/本地色调/本地结构/皮肤结构
→ 实时播放(带声音) → 框选屏幕区域实时渲染 → 导出 DLSS 视频/图片。
皮肤结构支持"双通道遮罩混合": 自动识别人物皮肤区域, 把皮肤结构增强限制在皮肤内。

【运行】
  双击 run.bat, 或:  python gui.py
  源码版依赖:  python -m pip install numpy opencv-python pillow sounddevice
  皮肤双通道(可选, 不装自动回退 OpenCV 肤色判据):  python -m pip install mediapipe
  播放声音 / 导出 CRF 画质(可选): 安装 ffmpeg 并加入 PATH, 不装则播放静音、
    导出回退 OpenCV 编码(CRF 不生效)。

【功能】
  - 实时预览: 原图 / DLSS / 左右对比(分界线可拖动), 参数调节立即生效。
  - 播放: 原帧率播放 + 同步声音(空格键播放/暂停), 音量/速度可调。
    速度 1.0 时自动降到 ≤720p 保实时; 低速(0.5/0.25)按原生分辨率渲染, 以慢放换满清。
  - 图片: png/jpg/bmp/tif/webp 单帧处理, 导出 <原名>_dlss.png。
  - 皮肤双通道[默认开]: 每帧跑两遍 DLSS(A 基准 + B 皮肤结构增强), 用 MediaPipe
    分割 + 人脸检测 + 肤色/纹理判据融合的皮肤遮罩把结构增量限制在皮肤内。
    代价: 帧率约减半; 未检出皮肤时自动退回单通道。
  - 肤色判据: 9 个滑条细调 + "一键自动调优"(按当前帧自动推导中心/容差),
    "叠加显示遮罩"可视化, 覆盖率实时显示。
  - 实时截屏测试: 框选屏幕区域实时 DLSS 渲染(DXGI 优先, GDI 兜底)。
  - 导出: 后台四级流水线(解码∥遮罩并行∥GPU 渲染∥写盘), 界面不卡顿。

【渲染库与 GPU 系列】(重要)
  nvngx_dlssnr.dll 与 GPU 系列强相关: 实测 50 系库在 40 系卡上直接起不来
  (0xBAD00001)。30/50 系显卡请使用对应系列的附加包:
    源码运行: 解压到本工具文件夹的【旁边】(与工具文件夹同级);
    EXE 版:   解压到 DLSS5Tool.exe 所在目录(与 _internal 同级)。
  启动时按显卡型号自动选用, 不匹配/不存在则回退自带的 40 系默认库(日志可见)。

【性能自适应】
  启动时按 GPU 型号/显存自动分档(旗舰/高/中/低/无GPU), 遮罩并行路数与复用
  步长随档位自适应; 导出中每 2s 采样 GPU 利用率动态升降档。无 NVIDIA GPU 时
  自动纯 CPU 回退渲染(OpenCV 锐化+色彩增强, 性能有限)。
  注意: 开皮肤双通道时瓶颈在 CPU 遮罩(MediaPipe)而非显卡——换更强的显卡不会
  明显更快; 想提速可关双通道或用中/低档(隔帧复用遮罩)。

【打包 EXE / 下载 EXE】
  现成的 EXE 版已上传本仓库 Releases(v1.0.0): DLSS5Tool-v1.0.0-win64.zip(下载 218MB,
  解压后约 440MB), 解压到任意目录双击 DLSS5Tool.exe 即用。

  自行打包: python -m pip install pyinstaller, 然后在工具目录运行:
    python -m PyInstaller --noconfirm --windowed --onedir --name DLSS5Tool
      --add-binary "dlssnr_host.dll;." --add-binary "nvngx_dlssnr.dll;."
      --add-data "models;models" --add-data "README.txt;."
      --collect-all mediapipe --exclude-module matplotlib gui.py
  (mediapipe 是函数内延迟 import, 必须 --collect-all 才打得进去)
  产物 dist\DLSS5Tool\ 零依赖: 对方电脑只需 Windows 10/11 + NVIDIA 驱动,
  整个文件夹一起拷贝, 别单独拽 exe。自检: "DLSS5Tool.exe --selftest" 应在 exe 旁
  生成 _selftest.txt 并含 DLSS_OK 与 SKIN_OK 两行。

【注意】
  - 混合显卡笔记本: 设置→屏幕→图形 里把 python.exe(EXE 版是 DLSS5Tool.exe)
    设为"高性能", 否则 DLSS 初始化失败(0xBAD00001)。
  - 换素材: 同尺寸复用 DLSS 会话, 不同尺寸自动重建 feature, 不会崩。
  - 源码版还需 VC++ 2015-2022 运行库(缺则装 Microsoft Visual C++ Redistributable)。
  - 本工具基于 NGX Feature 18 神经渲染(零引导), 不需要 torch/深度模型。
