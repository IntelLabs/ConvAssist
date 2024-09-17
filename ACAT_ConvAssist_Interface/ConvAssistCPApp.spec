# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# Increase recursion limit
sys.setrecursionlimit(5000)

# Define the data files to be included
datas = [
    ('ConvAssistCPApp/Assets', 'Assets'),
]

# Collect additional data files
datas += collect_data_files("en_core_web_sm")
datas += collect_data_files("sv_ttk")

import pkg_resources
for dist in pkg_resources.working_set:
    datas += copy_metadata(dist.project_name)


excludes=['mypy', 'pytest', 'pytest-cov', 'pytest-runner', 'pytest-xdist', 
        'pytest-forked', 'pytest-asyncio', 'pytest-astropy', 
        'pytest-doctestplus', 'pytest-openfiles', 'pytest-remotedata', 
        'pytest-doctestplus',
        'pyinstaller','pyinstaller-hooks-contrib','setuptools','pytest',
        'pytest-cov','pytest-cover','pytest-coverage','pytest-timeout',
        'pyinstaller-hooks-contrib','pyinstaller-hooks-contrib-hooks','pyinstaller-hooks-contrib-hooks-data',
        'pyinstaller', 'setuptools',
]

# Define the PyInstaller spec
a = Analysis( # type: ignore
    ['ConvAssistCPApp/ConvAssistUI.py'],  # Replace with your main script
    pathex=['../', 'ConvAssistCPApp'],  # Replace with your project path
    binaries=[],
    datas=datas,
    exclude=excludes,
    hiddenimports=[
        'en_core_web_sm', 
    ],
    hookspath=[],
    runtime_hooks=[],
    cipher=None,
    noarchive=True
)

# new_datas = []
# for d in a.datas:
#     if 'dist-info' in d[0]:
#         # new_datas.append(d)
#         print(d)

# a.datas = new_datas

# tqdmData = copy_metadata('tqdm')
# a.datas += tqdmData[0]


# # Uncomment the following lines if you need to include metadata for these packages
# a.datas.append(copy_metadata('tqdm'))
# # datas += copy_metadata('torch')
# # datas += copy_metadata('regex')
# # datas += copy_metadata('filelock')
# # datas += copy_metadata('packaging')
# # datas += copy_metadata('requests')
# # datas += copy_metadata('numpy')
# # datas += copy_metadata('tokenizers')
# # datas += copy_metadata('transformers')
# # datas += copy_metadata('huggingface-hub')
# # datas += copy_metadata('pyyaml')
# # datas += copy_metadata('blis')
# # datas += copy_metadata('certifi')
# # datas += copy_metadata('charset_normalizer')
# # datas += copy_metadata('cymem')
# # datas += copy_metadata('langcodes')
# # datas += copy_metadata('markupsafe')
# # datas += copy_metadata('murmurhash')

pyz = PYZ(a.pure) #type: ignore

exe = EXE( # type: ignore
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
	[('W ignore', None, 'OPTION')],
    name='ConvAssistApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ConvAssistCPApp/Assets/icon_tray.ico',
)

coll = COLLECT(exe, # type: ignore
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConvAssist-Dependencies'
)
               