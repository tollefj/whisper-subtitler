from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.speaker:
            result["speaker"] = self.speaker
        return result


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    audio_length_seconds: float
    coverage_percent: float

    @property
    def last_segment_end(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def is_complete(self) -> bool:
        return self.coverage_percent >= 95.0

    def to_dict(self) -> dict:
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "audio_length_seconds": self.audio_length_seconds,
            "coverage_percent": self.coverage_percent,
        }
