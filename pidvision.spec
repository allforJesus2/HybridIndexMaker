# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys
import os
sys.setrecursionlimit(10000)
block_cipher = None

# Collect all necessary data files and hidden imports
hidden_imports = [
    'easyocr',
    'torch',
    'torchvision',
    'detecto',
    'cv2',
    'PIL',
    'numpy',
    'pandas',
    'yaml',
    'openpyxl',
    'xlwings',
] + collect_submodules('easyocr')

# Add data files for easyocr
datas = collect_data_files('easyocr')

# Add additional data files for models and resources
datas += [
    ('models/*', 'models'),
    ('*.png', '.'),
    ('*.ico', '.'),
]

a = Analysis(
    ['PIDVision.py'],  # Replace with your main script name
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PIDVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='LOGO.ico'  # Make sure you have this icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PIDVision',
)