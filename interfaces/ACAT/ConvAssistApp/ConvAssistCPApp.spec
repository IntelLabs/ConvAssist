# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# Increase recursion limit
sys.setrecursionlimit(5000)

# Define the data files to be included
data = [
    ('ConvAssistCPApp/Assets', 'Assets'),
]

# Collect additional data files
data += collect_data_files("en_core_web_sm")
data+= collect_data_files("sv_ttk")

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
    data=data,
    exclude=excludes,
    hiddenimports=[
        'en_core_web_sm', 
    ],
    hookspath=[],
    runtime_hooks=[],
    cipher=None,
    noarchive=True
)
# remove_data = []
# for d in a.data:
#     if 'dist-info' in d[0]:
#         remove_data.append(d)

# for r in remove_data:
#     a.data.remove(r)

# a.data = new_data

# # Uncomment the following lines if you need to include metadata for these packages
# a.data += copy_metadata('tqdm')
# # data += copy_metadata('torch')
# # data += copy_metadata('regex')
# # data += copy_metadata('filelock')
# # data += copy_metadata('packaging')
# # data += copy_metadata('requests')
# # data += copy_metadata('numpy')
# # data += copy_metadata('tokenizers')
# # data += copy_metadata('transformers')
# # data += copy_metadata('huggingface-hub')
# # data += copy_metadata('pyyaml')
# # data += copy_metadata('blis')
# # data += copy_metadata('certifi')
# # data += copy_metadata('charset_normalizer')
# # data += copy_metadata('cymem')
# # data += copy_metadata('langcodes')
# # data += copy_metadata('markupsafe')
# # data += copy_metadata('murmurhash')

# print("Contents of a.zipfiles: ")
# for item in a.zipfiles:
#     print(item)

# print("Contents of a.pure:")
# for item in a.pure:
#     print(item)

# print("Contents of a.data:")
# for item in a.data:
#     print(item)

import pkg_resources
for dist in pkg_resources.working_set:
    data += copy_metadata(dist.project_name)

pyz = PYZ(a.pure) # type: ignore

exe = EXE( # type: ignore
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.data,
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
    a.data,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConvAssist-Dependencies'
)
               