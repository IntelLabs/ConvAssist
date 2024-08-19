from pathlib import Path
import time

def safe_delete_file(file_path):
    file_to_delete = Path(file_path)
    if file_to_delete.is_file():
        retries = 10

        for _ in range(retries):
            try:
                file_to_delete.unlink()
                break
            except PermissionError:
                time.sleep(.1)
                pass
            
    if file_to_delete.is_file():
        raise Exception(f"Could not delete file {file_path}")
