import logging
from pathlib import Path

import ffmpeg
import yt_dlp

from ..models import MediaSource, ProjectPaths

logger = logging.getLogger(__name__)


class MediaService:
    def prepare_media(
        self, source: MediaSource, paths: ProjectPaths, download_video: bool = False
    ) -> Path:
        if source.is_url:
            return self._handle_url(source, paths, download_video)
        return self._handle_local(source, paths)

    def _handle_url(
        self, source: MediaSource, paths: ProjectPaths, download_video: bool
    ) -> Path:
        logger.info(f"Downloading media from {source.path}")

        if download_video:
            self._download_video(source.path, paths.video_path)
            self._extract_audio(paths.video_path, paths.audio_path)
            return paths.video_path
        else:
            self._download_audio(source.path, paths.audio_path)
            return paths.audio_path

    def _handle_local(self, source: MediaSource, paths: ProjectPaths) -> Path:
        logger.info(f"Processing local file: {source.path}")
        video_path = Path(source.path)

        if not video_path.exists():
            raise FileNotFoundError(f"Media file not found: {source.path}")

        self._extract_audio(video_path, paths.audio_path)
        return video_path

    def _download_video(self, url: str, output: Path) -> None:
        max_resolution = "1080"
        parameters = {
            "format": f"bestvideo[ext=mp4][height<={max_resolution}]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(output),
        }

        with yt_dlp.YoutubeDL(parameters) as ydl:
            info = ydl.extract_info(url, download=True)
            logger.info(f"Downloaded: {info['title']}")

    def _download_audio(self, url: str, output: Path) -> None:
        output_without_ext = str(output).rsplit(".", 1)[0]
        parameters = {
            "format": "bestaudio/best",
            "outtmpl": output_without_ext,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
        }

        with yt_dlp.YoutubeDL(parameters) as ydl:
            info = ydl.extract_info(url, download=True)
            logger.info(f"Downloaded audio: {info['title']}")

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        logger.info(f"Extracting audio to {audio_path}")
        (
            ffmpeg.input(str(video_path))
            .output(str(audio_path), format="mp3", audio_bitrate="96k", vn=None)
            .overwrite_output()
            .run(quiet=True)
        )

    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        probe = ffmpeg.probe(str(audio_path))
        audio_stream = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "audio"
        )
        return float(audio_stream["duration"])
