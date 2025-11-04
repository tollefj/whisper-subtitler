import logging
from dataclasses import dataclass
from pathlib import Path

from .models import MediaSource, ProjectPaths, TranscriptionConfig, TranscriptionResult
from .services import MediaService, SubtitleService, TranscriptionService, VideoService

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    transcription: TranscriptionResult
    paths: ProjectPaths


class TranscriptionPipeline:
    def __init__(self, config: TranscriptionConfig, root_dir: Path):
        self.config = config
        self.root_dir = root_dir

        self.media_service = MediaService()
        self.transcription_service = TranscriptionService(config)
        self.subtitle_service = SubtitleService()
        self.video_service = VideoService()

    def run(
        self, media_path: str, save_video: bool = False
    ) -> PipelineResult:
        source = MediaSource.from_path(media_path)
        paths = ProjectPaths.create(self.root_dir, source.identifier)

        logger.info(f"Processing: {source.path}")
        logger.info(f"Output directory: {paths.store_dir}")

        video_path = self.media_service.prepare_media(
            source, paths, download_video=save_video
        )

        transcription = self.transcription_service.transcribe(paths)

        self.subtitle_service.generate_srt(transcription, paths.srt_path)

        if save_video:
            self._create_subtitled_video(video_path, paths)

        logger.info("Pipeline complete")
        return PipelineResult(transcription=transcription, paths=paths)

    def _create_subtitled_video(
        self, video_path: Path, paths: ProjectPaths
    ) -> None:
        logger.info("Creating subtitled video")
        self.video_service.burn_subtitles(
            video_path, paths.srt_path, paths.output_video_path
        )
        logger.info(f"Subtitled video saved: {paths.output_video_path}")
