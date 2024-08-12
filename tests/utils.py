from pathlib import Path

def safe_delete_file(file_path):
    file_to_delete = Path(file_path)
    if file_to_delete.is_file():
        file_to_delete.unlink()
