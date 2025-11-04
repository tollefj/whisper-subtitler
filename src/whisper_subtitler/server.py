import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from whisper_subtitler.models import TranscriptionConfig
from whisper_subtitler.services import ModelManager
from whisper_subtitler.services.job_queue import JobQueue, JobStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


shutdown_event = asyncio.Event()
job_queue: Optional[JobQueue] = None
model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue, model_manager

    root_dir = _get_root_dir()
    app_data_dir = _get_app_data_dir()

    logger.info(f"Root directory: {root_dir}")
    logger.info(f"App data directory: {app_data_dir}")

    model_manager = ModelManager(app_data_dir=app_data_dir)
    job_queue = JobQueue(root_dir=root_dir, model_manager=model_manager)

    logger.info("Server initialized successfully")

    yield

    logger.info("Shutting down server...")


app = FastAPI(title="Whisper Subtitler API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeRequest(BaseModel):
    media_path: str
    model: Optional[str] = "NbAiLab/nb-whisper-small"
    language: Optional[str] = "no"
    save_video: bool = False
    audio_only: bool = True


class JobResponse(BaseModel):
    id: str
    status: str
    progress: float
    message: str
    output_paths: Optional[dict] = None
    error: Optional[str] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    language: str
    size_mb: int
    is_bundled: bool
    is_downloaded: bool
    description: str


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": model_manager.get_optimal_device()[0] if model_manager else "unknown",
    }


@app.post("/transcribe", response_model=JobResponse)
async def create_transcription_job(request: TranscribeRequest):
    try:
        job = job_queue.create_job(request.media_path)

        config_path = _get_root_dir() / "config.yml"
        config = TranscriptionConfig.load(
            config_path=config_path,
            model_override=request.model,
            language_override=request.language,
        )

        asyncio.create_task(
            job_queue.submit_job(
                job.id,
                config=config,
                save_video=request.save_video and not request.audio_only,
            )
        )

        return JobResponse(
            id=job.id,
            status=job.status.value,
            progress=job.progress,
            message=job.message,
        )

    except Exception as e:
        logger.error(f"Failed to create transcription job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        id=job.id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        output_paths=job.output_paths,
        error=job.error,
    )


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs():
    jobs = job_queue.get_all_jobs()
    return [
        JobResponse(
            id=job.id,
            status=job.status.value,
            progress=job.progress,
            message=job.message,
            output_paths=job.output_paths,
            error=job.error,
        )
        for job in jobs
    ]


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    success = await job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or not active")
    return {"status": "cancelled"}


@app.delete("/jobs")
async def clear_completed_jobs():
    count = job_queue.clear_completed_jobs()
    return {"cleared": count}


@app.get("/models", response_model=list[ModelResponse])
async def list_models():
    models = model_manager.list_models()
    return [
        ModelResponse(
            id=model.id,
            name=model.name,
            language=model.language,
            size_mb=model.size_mb,
            is_bundled=model.is_bundled,
            is_downloaded=model.is_downloaded,
            description=model.description,
        )
        for model in models
    ]


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    model = model_manager.get_model_info(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelResponse(
        id=model.id,
        name=model.name,
        language=model.language,
        size_mb=model.size_mb,
        is_bundled=model.is_bundled,
        is_downloaded=model.is_downloaded,
        description=model.description,
    )


@app.post("/models/{model_id}/download")
async def download_model(model_id: str):
    try:
        if model_manager.is_model_available(model_id):
            return {"status": "already_downloaded"}

        path = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.download_model, model_id
        )

        return {"status": "downloaded", "path": str(path)}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to download model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/{job_id}")
async def stream_job_progress(job_id: str):
    async def event_generator():
        job = job_queue.get_job(job_id)
        if not job:
            yield f"data: {{'error': 'Job not found'}}\n\n"
            return

        last_progress = -1
        while job.status not in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            if job.progress != last_progress:
                yield f"data: {{'status': '{job.status.value}', 'progress': {job.progress}, 'message': '{job.message}'}}\n\n"
                last_progress = job.progress

            await asyncio.sleep(0.5)

        yield f"data: {{'status': '{job.status.value}', 'progress': {job.progress}, 'message': '{job.message}'}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown endpoint called")
    shutdown_event.set()
    return {"status": "shutting down"}


def _get_root_dir() -> Path:
    if getattr(sys, "_MEIPASS", None):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent.parent


def _get_app_data_dir() -> Path:
    app_data = os.getenv("APP_DATA_DIR")
    if app_data:
        return Path(app_data)

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "whisper-subtitler"
    elif sys.platform == "win32":
        return Path(os.getenv("APPDATA")) / "whisper-subtitler"
    else:
        return Path.home() / ".local" / "share" / "whisper-subtitler"


async def run_server():
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8008,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    await asyncio.wait(
        [server_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if shutdown_event.is_set():
        logger.info("Shutdown signal received, stopping server...")
        server.should_exit = True
        await server_task


def main():
    logger.info("Starting Whisper Subtitler API server...")
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
