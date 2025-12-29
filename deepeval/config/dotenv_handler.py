from dotenv import set_key, unset_key
from pathlib import Path


class DotenvHandler:
    def __init__(self, path: Path):
        self.path = Path(path)

    def upsert(self, mapping: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        for key, value in mapping.items():
            set_key(str(self.path), key, value, quote_mode="always")

    def unset(self, keys: set[str]) -> None:
        if not self.path.exists():
            return
        for key in keys:
            unset_key(str(self.path), key)
