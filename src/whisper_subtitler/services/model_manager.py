import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import whisperx

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    id: str
    name: str
    language: str
    size_mb: int
    is_bundled: bool
    is_downloaded: bool
    description: str


class ModelManager:
    MODELS = {
        "base": ModelInfo(
            id="base",
            name="Whisper Base",
            language="en",
            size_mb=140,
            is_bundled=True,
            is_downloaded=False,
            description="OpenAI Whisper base model (multilingual, good quality)",
        ),
        "nb-whisper-small": ModelInfo(
            id="NbAiLab/nb-whisper-small",
            name="Norwegian Whisper Small",
            language="no",
            size_mb=460,
            is_bundled=True,
            is_downloaded=False,
            description="Norwegian-optimized model (best for Norwegian)",
        ),
        "nb-whisper-medium": ModelInfo(
            id="NbAiLab/nb-whisper-medium",
            name="Norwegian Whisper Medium",
            language="no",
            size_mb=1400,
            is_bundled=False,
            is_downloaded=False,
            description="Norwegian-optimized model (higher accuracy, slower)",
        ),
        "nb-whisper-large": ModelInfo(
            id="NbAiLab/nb-whisper-large",
            name="Norwegian Whisper Large",
            language="no",
            size_mb=2800,
            is_bundled=False,
            is_downloaded=False,
            description="Norwegian-optimized model (highest accuracy, slowest)",
        ),
    }

    def __init__(self, app_data_dir: Optional[Path] = None):
        self.app_data_dir = app_data_dir or Path.home() / ".whisper-subtitler"
        self.models_dir = self.app_data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._setup_hf_cache()
        self._scan_models()

    def _setup_hf_cache(self) -> None:
        hf_cache = self.app_data_dir / "huggingface"
        hf_cache.mkdir(exist_ok=True)

        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache / "transformers")

        bundled_models = self._get_bundled_models_path()
        if bundled_models and bundled_models.exists():
            logger.info(f"Bundled models found at: {bundled_models}")

    def _get_bundled_models_path(self) -> Optional[Path]:
        if getattr(sys, "_MEIPASS", None):
            return Path(sys._MEIPASS) / "models"
        return Path(__file__).parent.parent.parent.parent / "models"

    def _scan_models(self) -> None:
        bundled_path = self._get_bundled_models_path()

        for key, model in self.MODELS.items():
            model_name_safe = model.id.replace("/", "--")
            downloaded_path = self.models_dir / model_name_safe

            if bundled_path:
                bundled_model_path = bundled_path / model_name_safe
                if bundled_model_path.exists():
                    model.is_downloaded = True
                    logger.info(f"Model {model.id} found in bundle")
                    continue

            if downloaded_path.exists():
                model.is_downloaded = True
                logger.info(f"Model {model.id} found in cache")

    def list_models(self) -> list[ModelInfo]:
        return list(self.MODELS.values())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.MODELS.values():
            if model.id == model_id or model.id.endswith(model_id):
                return model
        return None

    def is_model_available(self, model_id: str) -> bool:
        model = self.get_model_info(model_id)
        return model is not None and model.is_downloaded

    def get_model_path(self, model_id: str) -> Optional[Path]:
        model_name_safe = model_id.replace("/", "--")

        bundled_path = self._get_bundled_models_path()
        if bundled_path:
            bundled_model_path = bundled_path / model_name_safe
            if bundled_model_path.exists():
                return bundled_model_path

        downloaded_path = self.models_dir / model_name_safe
        if downloaded_path.exists():
            return downloaded_path

        return None

    def download_model(
        self, model_id: str, progress_callback=None
    ) -> Path:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")

        if model_info.is_downloaded:
            logger.info(f"Model {model_id} already downloaded")
            return self.get_model_path(model_id)

        logger.info(f"Downloading model {model_id} (~{model_info.size_mb}MB)...")

        if progress_callback:
            progress_callback(
                {"status": "downloading", "model": model_id, "progress": 0}
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model_name_safe = model_id.replace("/", "--")
        cache_dir = self.models_dir / model_name_safe
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = whisperx.load_model(
                model_id,
                device=device,
                compute_type=compute_type,
                download_root=str(self.models_dir),
            )

            if progress_callback:
                progress_callback(
                    {"status": "complete", "model": model_id, "progress": 100}
                )

            model_info.is_downloaded = True
            logger.info(f"Model {model_id} downloaded successfully")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return cache_dir

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            if progress_callback:
                progress_callback(
                    {
                        "status": "error",
                        "model": model_id,
                        "error": str(e),
                    }
                )
            raise

    def ensure_model_available(self, model_id: str) -> Path:
        if not self.is_model_available(model_id):
            return self.download_model(model_id)

        return self.get_model_path(model_id)

    def get_recommended_model(self, language: str = "no") -> str:
        if language == "no":
            return "NbAiLab/nb-whisper-small"
        return "base"

    def get_optimal_device(self) -> tuple[str, str]:
        if torch.cuda.is_available():
            return "cuda", "float16"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "float32"
        else:
            return "cpu", "int8"
