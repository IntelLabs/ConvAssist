# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# Increase recursion limit
sys.setrecursionlimit(5000)

# Define the data files to be included
datas = [
    ('Assets', '.'),
    ('Assets/icon_tray.png', 'Assets'),
    ('Assets/button_back.png', 'Assets'),
    ('Assets/button_clear.png', 'Assets'),
    ('Assets/button_exit.png', 'Assets'),
    ('Assets/button_license.png', 'Assets'),
    ('Assets/frame.png', 'Assets'),
    ('scipy.libs', '.'),
    ('scipy.libs/libopenblas-802f9ed1179cb9c9b03d67ff79f48187.dll', 'scipy.libs')
]

# Collect additional data files
datas += collect_data_files("en_core_web_sm")

# Uncomment the following lines if you need to include metadata for these packages
# datas += copy_metadata('tqdm')
# datas += copy_metadata('torch')
# datas += copy_metadata('regex')
# datas += copy_metadata('filelock')
# datas += copy_metadata('packaging')
# datas += copy_metadata('requests')
# datas += copy_metadata('numpy')
# datas += copy_metadata('tokenizers')
# datas += copy_metadata('transformers')
# datas += copy_metadata('huggingface-hub')
# datas += copy_metadata('pyyaml')

# Define the PyInstaller spec
a = Analysis(
    ['ConvAssistCPApp.py'],  # Replace with your main script
    pathex=['.'],
    binaries=[],
    datas=datas,
    # hiddenimports=[],
    hiddenimports=['en_core_web_sm', 
                   'huggingface_hub.hf_api',
                   'huggingface_hub.repository', 
                   'torch', 
                   'tqdm', 
                   'scipy.datasets', 
                   'scipy.fftpack', 
                   'scipy.misc', 
                   'scipy.odr', 
                   'scipy.signal', 
                   'sklearn.utils._typedefs', 
                   'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
                   'sklearn.metrics._pairwise_distances_reduction._middle_term_computer', 
                   'sklearn.utils._heap', 
                   'sklearn.utils._sorting',
                   'sklearn.utils._vector_sentinel'
                ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ConvAssistCPApp',  # Replace with your executable name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True  # Set to False if you want to create a windowed application
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConvAssistCPApp'  # Replace with your executable name
)