import logging
import time
from pathlib import Path

from ..models import Segment, TranscriptionResult

logger = logging.getLogger(__name__)


class SubtitleService:
    @staticmethod
    def generate_srt(result: TranscriptionResult, output_path: Path) -> None:
        logger.info(f"Generating SRT file: {output_path}")

        srt_content = SubtitleService._create_srt_content(result.segments)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT file saved with {len(result.segments)} segments")

    @staticmethod
    def _create_srt_content(segments: list[Segment]) -> str:
        entries = []

        for index, segment in enumerate(segments, start=1):
            start_time = SubtitleService._format_timestamp(segment.start)
            end_time = SubtitleService._format_timestamp(segment.end)

            text = segment.text
            if segment.speaker:
                text = f"{segment.speaker}: {text}"

            entry = f"{index}\n{start_time} --> {end_time}\n{text}\n\n"
            entries.append(entry)

        return "".join(entries)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        return time.strftime("%H:%M:%S,000", time.gmtime(seconds))
