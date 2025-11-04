import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from ..models import TranscriptionConfig, TranscriptionResult
from ..pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TranscriptionJob:
    id: str
    media_path: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[TranscriptionResult] = None
    output_paths: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class JobQueue:
    def __init__(self, root_dir: Path, model_manager):
        self.root_dir = root_dir
        self.model_manager = model_manager
        self.jobs: dict[str, TranscriptionJob] = {}
        self.active_tasks: dict[str, asyncio.Task] = {}

    def create_job(self, media_path: str) -> TranscriptionJob:
        job_id = str(uuid.uuid4())
        job = TranscriptionJob(id=job_id, media_path=media_path)
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} for {media_path}")
        return job

    def get_job(self, job_id: str) -> Optional[TranscriptionJob]:
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list[TranscriptionJob]:
        return list(self.jobs.values())

    async def submit_job(
        self,
        job_id: str,
        config: TranscriptionConfig,
        save_video: bool = False,
    ) -> TranscriptionJob:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        task = asyncio.create_task(
            self._process_job(job, config, save_video)
        )
        self.active_tasks[job_id] = task

        return job

    async def _process_job(
        self,
        job: TranscriptionJob,
        config: TranscriptionConfig,
        save_video: bool,
    ) -> None:
        try:
            self._ensure_model_available(job, config.effective_model)

            self._update_job(job, JobStatus.DOWNLOADING, 10, "Preparing media...")
            await asyncio.sleep(0.1)

            pipeline = TranscriptionPipeline(config=config, root_dir=self.root_dir)

            self._update_job(job, JobStatus.EXTRACTING, 20, "Extracting audio...")
            await asyncio.sleep(0.1)

            self._update_job(job, JobStatus.TRANSCRIBING, 30, "Transcribing...")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, pipeline.run, job.media_path, save_video
            )

            self._update_job(job, JobStatus.GENERATING, 90, "Generating subtitles...")
            await asyncio.sleep(0.1)

            job.result = result.transcription
            job.output_paths = result.paths.to_dict()

            self._update_job(job, JobStatus.COMPLETED, 100, "Transcription complete")
            job.completed_at = datetime.now()

            logger.info(f"Job {job.id} completed successfully")

        except asyncio.CancelledError:
            self._update_job(job, JobStatus.CANCELLED, job.progress, "Job cancelled")
            logger.info(f"Job {job.id} cancelled")
            raise

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Job {job.id} failed: {error_msg}", exc_info=True)
            job.error = error_msg
            self._update_job(job, JobStatus.FAILED, job.progress, f"Error: {error_msg}")
            job.completed_at = datetime.now()

        finally:
            if job.id in self.active_tasks:
                del self.active_tasks[job.id]

    def _ensure_model_available(self, job: TranscriptionJob, model_id: str) -> None:
        if not self.model_manager.is_model_available(model_id):
            self._update_job(
                job,
                JobStatus.DOWNLOADING,
                5,
                f"Downloading model {model_id}...",
            )

            self.model_manager.download_model(
                model_id,
                progress_callback=lambda info: self._update_job(
                    job,
                    JobStatus.DOWNLOADING,
                    5 + (info.get("progress", 0) * 0.05),
                    f"Downloading model: {info.get('progress', 0)}%",
                ),
            )

    def _update_job(
        self, job: TranscriptionJob, status: JobStatus, progress: float, message: str
    ) -> None:
        job.status = status
        job.progress = progress
        job.message = message

    async def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False

        if job_id in self.active_tasks:
            task = self.active_tasks[job_id]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            return True

        return False

    def clear_completed_jobs(self) -> int:
        completed_ids = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        ]

        for job_id in completed_ids:
            del self.jobs[job_id]

        logger.info(f"Cleared {len(completed_ids)} completed jobs")
        return len(completed_ids)
