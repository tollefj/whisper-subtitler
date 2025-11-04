import gc
import json
import logging
import warnings
from pathlib import Path

import torch
import whisperx

from ..models import ProjectPaths, Segment, TranscriptionConfig, TranscriptionResult

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class TranscriptionService:
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_enabled = torch.cuda.is_available()

    def transcribe(self, paths: ProjectPaths) -> TranscriptionResult:
        audio = whisperx.load_audio(str(paths.audio_path))
        audio_length = len(audio) / SAMPLE_RATE

        raw_result = self._get_or_create_transcription(audio, paths)
        aligned_result = self._align_transcription(raw_result, audio)

        segments = self._extract_segments(aligned_result)
        coverage = self._calculate_coverage(segments, audio_length)

        result = TranscriptionResult(
            segments=segments,
            audio_length_seconds=audio_length,
            coverage_percent=coverage,
        )

        self._log_coverage(result)
        return result

    def _get_or_create_transcription(
        self, audio, paths: ProjectPaths
    ) -> dict:
        if paths.raw_transcript_path.exists():
            logger.info("Loading cached transcription")
            with open(paths.raw_transcript_path) as f:
                return json.load(f)

        logger.info("Starting transcription")
        model = whisperx.load_model(
            self.config.effective_model,
            device=self.device,
            compute_type=self.config.whisper.compute_type,
            language=self.config.effective_language,
        )

        result = model.transcribe(audio, batch_size=self.config.whisper.batch_size)

        self._cache_transcription(result, paths)
        self._cleanup_model(model)

        return result

    def _cache_transcription(self, result: dict, paths: ProjectPaths) -> None:
        logger.info("Caching transcription")
        with open(paths.raw_transcript_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def _align_transcription(self, result: dict, audio) -> dict:
        logger.info("Aligning transcription")

        alignment_model, metadata = whisperx.load_align_model(
            language_code=self.config.effective_language,
            device=self.device,
        )

        aligned = whisperx.align(
            result["segments"],
            alignment_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        self._cleanup_model(alignment_model)
        return aligned

    def _extract_segments(self, result: dict) -> list[Segment]:
        return [
            Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                speaker=seg.get("speaker"),
            )
            for seg in result["segments"]
        ]

    def _calculate_coverage(
        self, segments: list[Segment], audio_length: float
    ) -> float:
        if not segments:
            return 0.0

        last_end = segments[-1].end
        return (last_end / audio_length) * 100

    def _log_coverage(self, result: TranscriptionResult) -> None:
        logger.info(f"Audio length: {result.audio_length_seconds:.2f}s")
        logger.info(f"Last segment ends at: {result.last_segment_end:.2f}s")
        logger.info(f"Coverage: {result.coverage_percent:.1f}%")

        if not result.is_complete:
            missing = result.audio_length_seconds - result.last_segment_end
            logger.warning(
                f"Incomplete transcription: {result.coverage_percent:.1f}% "
                f"(missing {missing:.2f}s)"
            )

    def _cleanup_model(self, model) -> None:
        if self.cuda_enabled:
            gc.collect()
            torch.cuda.empty_cache()
        del model
