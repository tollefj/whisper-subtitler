from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    store_dir: Path
    video_path: Path
    audio_path: Path
    srt_path: Path
    raw_transcript_path: Path
    output_video_path: Path

    @staticmethod
    def create(root: Path, identifier: str) -> "ProjectPaths":
        store_dir = root / "store" / identifier
        store_dir.mkdir(parents=True, exist_ok=True)

        return ProjectPaths(
            root=root,
            store_dir=store_dir,
            video_path=store_dir / "video.mp4",
            audio_path=store_dir / "audio.mp3",
            srt_path=store_dir / "subtitles.srt",
            raw_transcript_path=store_dir / "raw_transcript.json",
            output_video_path=store_dir / f"{identifier}_subtitled.mp4",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "video": str(self.video_path),
            "audio": str(self.audio_path),
            "srt": str(self.srt_path),
            "raw_transcript": str(self.raw_transcript_path),
        }
