from pathlib import Path


def represents_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_data_size(path: Path):
    data_size = 0
    for _ in path.iterdir():
        data_size += 1
    return data_size
