from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WhisperConfig:
    model: str
    batch_size: int
    compute_type: str
    language: str

    @staticmethod
    def from_dict(data: dict) -> "WhisperConfig":
        return WhisperConfig(
            model=data.get("model", "base"),
            batch_size=data.get("batch_size", 8),
            compute_type=data.get("compute_type", "int8"),
            language=data.get("language", "no"),
        )


@dataclass(frozen=True)
class TranscriptionConfig:
    whisper: WhisperConfig
    model_override: Optional[str]
    language_override: Optional[str]

    @property
    def effective_model(self) -> str:
        return self.model_override or self.whisper.model

    @property
    def effective_language(self) -> str:
        return self.language_override or self.whisper.language

    @staticmethod
    def load(
        config_path: Path,
        model_override: Optional[str] = None,
        language_override: Optional[str] = None,
    ) -> "TranscriptionConfig":
        import yaml

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        whisper_config = WhisperConfig.from_dict(config_data.get("whisper", {}))

        return TranscriptionConfig(
            whisper=whisper_config,
            model_override=model_override,
            language_override=language_override,
        )
