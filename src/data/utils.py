from pathlib import Path


def represents_int(s: str) -> int:
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_data_size(path: Path) -> int:
    data_size = 0
    for entry in path.iterdir():
        if represents_int(entry.name.removesuffix(entry.suffix)):
            data_size += 1
    return data_size
