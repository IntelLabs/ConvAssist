from pathlib import Path
import time

def safe_delete_file(file_path):
    file_to_delete = Path(file_path)
    if file_to_delete.is_file():
        retries = 5

        for _ in range(retries):
            try:
                file_to_delete.unlink()
                break
            except PermissionError:
                time.sleep(.1)
                pass
