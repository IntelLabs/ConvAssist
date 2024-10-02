# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import time
from pathlib import Path


def safe_delete_file(file_path):
    file_to_delete = Path(file_path)
    if file_to_delete.is_file():
        retries = 10

        for _ in range(retries):
            try:
                file_to_delete.unlink()
                break
            except PermissionError:
                time.sleep(0.1)
                pass

    if file_to_delete.is_file():
        raise Exception(f"Could not delete file {file_path}")


def safe_check_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
