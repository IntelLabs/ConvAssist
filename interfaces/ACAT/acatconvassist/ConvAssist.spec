# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import copy_metadata, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

SCRIPT_DIR = os.path.dirname(os.path.realpath(__name__))

additionaldata = []
additionaldata = [('assets', 'assets')]
additionaldata += copy_metadata('convassist')
additionaldata += collect_data_files("sv_ttk")
additionaldata += collect_data_files("en_core_web_sm")
additionaldata += collect_data_files("spellchecker")

print(f'script dir: {SCRIPT_DIR}')
print(f'additional data: {additionaldata}')

a = Analysis(
    [f'{SCRIPT_DIR}\\ConvAssist.py'],
    pathex=[],
    binaries=[],
    datas=additionaldata,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tzdata','pytz'],
    noarchive=False,
    optimize=0,
)


pyz = PYZ(a.pure, a.zipped_data)


exe = EXE(
    pyz,
    a.scripts,
    # a.binaries,
    a.zipfiles,
    a.datas,
    [('W ignore', None, 'OPTION')],
    exclude_binaries=True,
    name='ConvAssist',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=f'{SCRIPT_DIR}\\Assets\\icon_tray.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    excludes=['_include/transformers/models/deprecated'],
    name='ConvAssist',
)