# -*- mode: python ; coding: utf-8 -*-
import sys
import pkg_resources
sys.setrecursionlimit(5000)
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files

additionaldata = [('Assets', 'Assets')]

# # Iterate over all installed packages and collect metadata
# for dist in pkg_resources.working_set:
#     additionaldata += copy_metadata(dist.project_name)
#     additionaldata += collect_data_files(dist.project_name)

# additionaldata += copy_metadata('convassist')
# additionaldata += copy_metadata('tqdm')
# additionaldata += copy_metadata('torch')
# additionaldata += copy_metadata('regex')
# additionaldata += copy_metadata('filelock')
# additionaldata += copy_metadata('packaging')
# additionaldata += copy_metadata('requests')
# additionaldata += copy_metadata('numpy')
# additionaldata += copy_metadata('tokenizers')
# additionaldata += copy_metadata('transformers')
# additionaldata += copy_metadata('huggingface-hub')
# additionaldata += copy_metadata('pyyaml')

additionaldata += collect_data_files("en_core_web_sm")
additionaldata += collect_data_files("sv-ttk")

# hiddenimports=[
#     'en_core_web_sm', 
#     'huggingface_hub.hf_api', 
#     'huggingface_hub.repository', 
#     'torch',
#     'tqdm', 
#     'scipy.datasets', 
#     'scipy.fftpack', 
#     'scipy.misc', 
#     'scipy.odr', 
#     'scipy.signal', 
#     'sklearn.utils._typedefs', 
#     'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
#     'sklearn.metrics._pairwise_distances_reduction._middle_term_computer', 
#     'sklearn.utils._heap', 
#     'sklearn.utils._sorting',
#     'sklearn.utils._vector_sentinel'
# ]

a = Analysis( # type: ignore
    ['ConvAssistUI.py'],
    pathex=[],
    binaries=[],
    datas=additionaldata,
    # hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # excludes=['torch.onnx', 'scipy.special', 'torch.tensorboard', 'scipy.special.cython_special'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None) # type: ignore

exe = EXE( # type: ignore
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
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon_tray.ico',
)
coll = COLLECT( # type: ignore
    exe,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConvAssistUI',
)
