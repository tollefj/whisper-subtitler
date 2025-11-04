from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MediaType(Enum):
    URL = "url"
    LOCAL = "local"


@dataclass(frozen=True)
class MediaSource:
    path: str
    media_type: MediaType

    @staticmethod
    def from_path(path: str) -> "MediaSource":
        is_url = any(indicator in path for indicator in ["http", "www", "youtube"])
        media_type = MediaType.URL if is_url else MediaType.LOCAL
        return MediaSource(path=path, media_type=media_type)

    @property
    def is_url(self) -> bool:
        return self.media_type == MediaType.URL

    @property
    def is_local(self) -> bool:
        return self.media_type == MediaType.LOCAL

    @property
    def identifier(self) -> str:
        if self.is_url:
            return self.path.split("/")[-1].split("?")[0]
        return Path(self.path).stem
