import gc
import json
import logging
import os
import time
import warnings
from typing import Dict

from torch import cuda

# avoid PyAnnote warnings for torchaudio backend.
warnings.filterwarnings("ignore", category=UserWarning)
import whisperx
from whisperx.diarize import DiarizationPipeline


def whisperx_transcription(
    store,
    config: Dict[str, str],
    diarize_config: Dict[str, str] = None,
    model_id=None,
    min_speakers=None,
    max_speakers=None,
    language="no",
):
    logger = logging.getLogger(__name__)
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logger.info(locals())
    cuda_enabled = cuda.is_available()
    device = "cuda" if cuda_enabled else "cpu"
    model_id = model_id if model_id else config["model"]

    audio = whisperx.load_audio(store["audio"])

    if os.path.exists(store["raw_transcript"]):
        logger.info("Found cached raw transcription, loading...")
        with open(store["raw_transcript"], "r") as f:
            result = json.load(f)
    else:
        model = whisperx.load_model(
            model_id,
            device=device,
            compute_type=config["compute_type"],
            language=language,
        )

        logger.info("1. Transcribing...")
        result = model.transcribe(audio, batch_size=config["batch_size"])

        logger.info("Caching raw transcription...")
        with open(store["raw_transcript"], "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if cuda_enabled:
            gc.collect()
            cuda.empty_cache()
        del model

    logger.info("2. Aligning...")

    alignment, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        alignment,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if cuda_enabled:
        gc.collect()
        cuda.empty_cache()
    del alignment

    if diarize_config:
        logger.info("3. Diarizing...")
        diarize_model = DiarizationPipeline(
            use_auth_token=diarize_config["HF"], device=device
        )
        diarize_segments = diarize_model(
            audio, min_speakers=min_speakers, max_speakers=max_speakers
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

    length_in_seconds = len(audio) / whisperx.audio.SAMPLE_RATE
    logger.info(f"Audio length: {length_in_seconds:.2f} seconds")

    if result["segments"]:
        last_segment_end = result["segments"][-1]["end"]
        logger.info(f"Last segment ends at: {last_segment_end:.2f} seconds")
        coverage_percent = (last_segment_end / length_in_seconds) * 100
        logger.info(f"Transcription coverage: {coverage_percent:.1f}%")

        if coverage_percent < 95:
            logger.warning(
                f"Transcription incomplete: only {coverage_percent:.1f}% of audio transcribed. "
                f"Missing {length_in_seconds - last_segment_end:.2f} seconds."
            )
    else:
        logger.error("No segments found in transcription")

    return result


def create_segment(index, start, end, text, speaker=None):
    start = time.strftime("%H:%M:%S,000", time.gmtime(start))
    end = time.strftime("%H:%M:%S,000", time.gmtime(end))
    text = text.strip()
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
    }


def segments_to_srt(segments, output=None):
    subs = []
    for i, segment in enumerate(segments):
        start = segment["start"]
        end = segment["end"]
        start = time.strftime("%H:%M:%S,000", time.gmtime(start))
        end = time.strftime("%H:%M:%S,000", time.gmtime(end))
        text = segment["text"].strip()
        speaker = segment.get("speaker")
        text = f"{speaker}: {text}" if speaker else text
        sub = f"{i + 1}\n{start} --> {end}\n{text}\n\n"
        subs.append(sub)

    if output:
        with open(output, "w") as f:
            f.writelines(subs)
    return subs
