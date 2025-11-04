import logging
from pathlib import Path
from sys import platform

import ffmpeg
import torch

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self):
        self.video_codec = self._determine_codec()

    def burn_subtitles(
        self, video_path: Path, subtitle_path: Path, output_path: Path
    ) -> None:
        logger.info(f"Burning subtitles into video: {output_path}")
        logger.info(f"Using codec: {self.video_codec}")

        (
            ffmpeg.input(str(video_path))
            .output(
                str(output_path),
                vcodec=self.video_codec,
                acodec="aac",
                vf=f"subtitles={subtitle_path}",
                ar=48000,
                max_muxing_queue_size=9999,
                pix_fmt="yuv420p",
                maxrate="1500k",
                bufsize="2000k",
            )
            .overwrite_output()
            .run(quiet=True)
        )

        logger.info("Subtitle burning complete")

    @staticmethod
    def _determine_codec() -> str:
        if torch.cuda.is_available():
            return "h264_nvenc"
        elif platform == "darwin":
            return "h264_videotoolbox"
        return "libx264"
