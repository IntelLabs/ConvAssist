# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import copy_metadata, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

SCRIPT_DIR = os.path.dirname(os.path.realpath(__name__))

additionaldata = []
# additionaldata = [('src\\assets', 'assets')]
additionaldata += copy_metadata('convassist')
additionaldata += collect_data_files("sv_ttk")
additionaldata += collect_data_files("en_core_web_sm")

print(f'script dir: {SCRIPT_DIR}')
print(f'additional data: {additionaldata}')

a = Analysis(
    [f'{SCRIPT_DIR}\\ConvAssistUI.py'],
    pathex=[],
    binaries=[],
    datas=additionaldata,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)


pyz = PYZ(a.pure, a.zipped_data)


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [('W ignore', None, 'OPTION')],
    name='ConvAssistUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=f'{SCRIPT_DIR}\\src\\assets\\icon_tray.ico'
)


coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConvAssistUI',
)
