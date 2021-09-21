from pathlib import Path


def represents_int(s: str) -> int:
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_data_size(path: Path) -> int:
    data_size = 0
    for _ in path.iterdir():
        data_size += 1
    return data_size
