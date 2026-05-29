# -*- mode: python ; coding: utf-8 -*-
import os, sys
from PyInstaller.utils.hooks import collect_all

# 当前 Python 版本标签（如 cp310）
PYTHON_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"

datas = [('config.yaml', '.')]
binaries = []
hiddenimports = [
    'src', 'src.gui', 'src.ball_detector', 'src.ball_tracker',
    'src.tracking', 'src.viscosity', 'src.velocity', 'src.terminal_region',
    'src.terminal_velocity', 'src.plotting', 'src.report', 'src.utils',
    'src.video_io', 'src.roi_manager', 'src.background_model',
    'src.candidate_detector', 'src.interval_selector', 'src.trajectory_filter',
    'src.ball_geometry', 'src.pipeline',
]
tmp_ret = collect_all('PySide6')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Viscometer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Viscometer',
)
