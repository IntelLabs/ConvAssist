# ------------------------------------------------------------------
# Copyright (c) 2020 PyInstaller Development Team.
#
# This file is distributed under the terms of the GNU General Public
# License (version 2.0 or later).
#
# The full license is available in LICENSE, distributed with
# this software.
#
# SPDX-License-Identifier: GPL-2.0-or-later
# ------------------------------------------------------------------

import os

from PyInstaller.utils.hooks import (
    logger,
    collect_data_files,
    is_module_satisfies,
    collect_dynamic_libs,
    collect_submodules,
    get_package_paths,
)

if is_module_satisfies("PyInstaller >= 6.0"):
    from PyInstaller.compat import is_linux, is_win
    from PyInstaller.utils.hooks import PY_DYLIB_PATTERNS

    module_collection_mode = "pyz+py"
    warn_on_missing_hiddenimports = False

    datas = collect_data_files(
        "torch",
        excludes=[
            "**/*.h",
            "**/*.hpp",
            "**/*.cuh",
            "**/*.lib",
            "**/*.cpp",
            "**/*.pyi",
            "**/*.cmake",
        ],
    )

    include_modules = [
        `"torch",
        "torch.xpu",
        "torch.version",
        "torch.utils",
        "torch.types",
        "torch.torch_version",
        "torch.testing",
        "torch.storage",
        "torch.special",
        "torch.sparse",
        "torch.signal", 
        "torch.serialization",
        "torch.return_types",
        "torch.random",
        "torch.quasirandom",
        "torch.quantization",
        "torch.profiler",
        "torch.package",
        "torch.overrides",
        "torch.optim",
        "torch.onnx",
        "torch.nn",
        "torch.nested",
        "torch.multiprocessing",
        "torch.mtia",
        "torch.mps",
        "torch.masked",
        "torch.linalg",
        "torch.library",
        "torch.jit",
        "torch.hub",
        "torch.fx",
        "torch.futures",
        "torch.functional",
        "torch.func",
        "torch.fft",
        "torch.export",
        "torch.distributions",
        "torch.distributed",
        "torch.cuda",
        "torch.cpu",
        "torch.compiler",
        "torch.backends",
        "torch.autograd",
        "torch.ao",
        "torch.amp",
        "torch._weights_only_unpickler",
        "torch._vmap_internals",
        "torch._VF",
        "torch._vendor",
        "torch._utils",
        "torch._utils_internal",
        "torch._torch_docs",
        "torch._tensor",
        "torch._tensor_str",
        "torch._tensor_docs",
        "torch._subclasses",
        "torch._strobelight",
        "torch._streambase",
        "torch._storage_docs",
        "torch._sources",
        "torch._size_docs",
        "torch._refs",
        "torch._prims",
        "torch._prims_common",
        "torch._ops",
        "torch._numpy",
        "torch._namedtensor_internals",
        "torch._meta_registrations",
        "torch._lowrank",
        "torch._logging",
        "torch._lobpcg",
        "torch._linalg_utils",
        "torch._library",
        "torch._jit_internal",
        "torch._inductor",
        "torch._higher_order_ops",
        "torch._guards",
        "torch._functorch",
        "torch._dynamo",
        "torch._dispatch",
        "torch._decomp",
        "torch._custom_op",
        "torch._compile",
        "torch._classes",
        "torch._C",
        "torch._awaits",
        "torch.__future__",
        "torch.__config__",
    ]

    hiddenimports = collect_submodules("torch", filter=lambda name: any(name.startswith(module) for module in include_modules))


    binaries = collect_dynamic_libs(
        "torch",
        # Ensure we pick up fully-versioned .so files as well
        search_patterns=PY_DYLIB_PATTERNS + ['*.so.*'],
    )

    # On Linux, torch wheels built with non-default CUDA version bundle CUDA libraries themselves (and should be handled
    # by the above `collect_dynamic_libs`). Wheels built with default CUDA version (which are available on PyPI), on the
    # other hand, use CUDA libraries provided by nvidia-* packages. Due to all possible combinations (CUDA libs from
    # nvidia-* packages, torch-bundled CUDA libs, CPU-only CUDA libs) we do not add hidden imports directly, but instead
    # attempt to infer them from requirements listed in the `torch` metadata.
    if is_linux:
        def _infer_nvidia_hiddenimports():
            import packaging.requirements
            from _pyinstaller_hooks_contrib.compat import importlib_metadata
            from _pyinstaller_hooks_contrib.utils import nvidia_cuda as cudautils

            dist = importlib_metadata.distribution("torch")
            requirements = [packaging.requirements.Requirement(req) for req in dist.requires or []]
            requirements = [req.name for req in requirements if req.marker is None or req.marker.evaluate()]

            return cudautils.infer_hiddenimports_from_requirements(requirements)

        try:
            nvidia_hiddenimports = _infer_nvidia_hiddenimports()
        except Exception:
            # Log the exception, but make it non-fatal
            logger.warning("hook-torch: failed to infer NVIDIA CUDA hidden imports!", exc_info=True)
            nvidia_hiddenimports = []
        logger.info("hook-torch: inferred hidden imports for CUDA libraries: %r", nvidia_hiddenimports)
        hiddenimports += nvidia_hiddenimports

    # The Windows nightly build for torch 2.3.0 added dependency on MKL. The `mkl` distribution does not provide an
    # importable package, but rather installs the DLLs in <env>/Library/bin directory. Therefore, we cannot write a
    # separate hook for it, and must collect the DLLs here. (Most of these DLLs are missed by PyInstaller's binary
    # dependency analysis due to being dynamically loaded at run-time).
    if is_win:
        def _collect_mkl_dlls():
            import packaging.requirements
            from _pyinstaller_hooks_contrib.compat import importlib_metadata

            # Check if torch depends on `mkl`
            dist = importlib_metadata.distribution("torch")
            requirements = [packaging.requirements.Requirement(req) for req in dist.requires or []]
            requirements = [req.name for req in requirements if req.marker is None or req.marker.evaluate()]
            if 'mkl' not in requirements:
                logger.info('hook-torch: this torch build does not depend on MKL...')
                return []  # This torch build does not depend on MKL

            # Find requirements of mkl - this should yield `intel-openmp` and `tbb`, which install DLLs in the same
            # way as `mkl`.
            try:
                dist = importlib_metadata.distribution("mkl")
            except importlib_metadata.PackageNotFoundError:
                return []  # For some reason, `mkl` distribution is unavailable.
            requirements = [packaging.requirements.Requirement(req) for req in dist.requires or []]
            requirements = [req.name for req in requirements if req.marker is None or req.marker.evaluate()]

            requirements = ['mkl'] + requirements

            mkl_binaries = []
            logger.info('hook-torch: collecting DLLs from MKL and its dependencies: %r', requirements)
            for requirement in requirements:
                try:
                    dist = importlib_metadata.distribution(requirement)
                except importlib_metadata.PackageNotFoundError:
                    continue

                # Go over files, and match DLLs in <env>/Library/bin directory
                for dist_file in dist.files:
                    if not dist_file.match('../../Library/bin/*.dll'):
                        continue
                    dll_file = dist.locate_file(dist_file).resolve()
                    mkl_binaries.append((str(dll_file), '.'))

            logger.info(
                'hook-torch: found MKL DLLs: %r',
                sorted([os.path.basename(src_name) for src_name, dest_name in mkl_binaries])
            )
            return mkl_binaries

        try:
            mkl_binaries = _collect_mkl_dlls()
        except Exception:
            # Log the exception, but make it non-fatal
            logger.warning("hook-torch: failed to collect MKL DLLs!", exc_info=True)
            mkl_binaries = []
        binaries += mkl_binaries
else:
    datas = [(get_package_paths("torch")[1], "torch")]

# With torch 2.0.0, PyInstaller's modulegraph analysis hits the recursion limit.
# So, unless the user has already done so, increase it automatically.
if is_module_satisfies("torch >= 2.0.0"):
    import sys

    new_limit = 5000
    if sys.getrecursionlimit() < new_limit:
        logger.info("hook-torch: raising recursion limit to %d", new_limit)
        sys.setrecursionlimit(new_limit)
