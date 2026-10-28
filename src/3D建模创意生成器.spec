# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置 —— 3D 建模创意生成器 v2。

单文件（onefile）模式：产出一个可直接双击运行的 exe。
依赖只有 numpy + Pillow，体积可控；tkinter 由 PyInstaller 自动收编。
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
datas += collect_data_files("creative3d", include_py_files=False)

# tkinter 的 tcl/tk 运行时由 PyInstaller 自动处理；这里只补我们自己的资源
datas += [("assets/icon.ico", "assets"), ("assets/icon.png", "assets")]

hiddenimports = collect_submodules("creative3d")
hiddenimports += [
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFilter",
    "PIL.ImageGrab",
    "PIL.ImageTk",
    "tkinter",
    "tkinter.ttk",
    "tkinter.font",
]

excludes = [
    "matplotlib",
    "scipy",
    "pandas",
    "pytest",
    "PyQt5",
    "PySide2",
    "PySide6",
    "IPython",
    "notebook",
    "torch",
    "tensorflow",
    "cv2",
    "sklearn",
]

block_cipher = None

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="3D建模创意生成器",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico",
    version=None,
)
