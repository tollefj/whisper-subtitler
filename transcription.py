import argparse
import logging
import os

import yaml

from utils.ffmpeg_util import extract_audio, write_video_with_subs
from utils.whisper_util import segments_to_srt, whisperx_transcription
from utils.youtube_util import download_audio, download_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "output")


ext = (".mp4", ".mkv", ".avi")


logging.basicConfig(level=logging.INFO)


def transcribe(
    media_path: str,
    model: str = "NbAiLab/nb-whisper-small",
    diarize: bool = False,
    save: bool = False,  # saves the subtitled video in the output folder
    language: str = "no",
) -> None:
    config = yaml.load(open("config.yml"), Loader=yaml.FullLoader)

    name = media_path.split("/")[-1].split(".")[0]
    tmp_folder = os.path.join(BASE_DIR, "store", name)
    os.makedirs(tmp_folder, exist_ok=True)
    STORE = {
        "video": os.path.join(tmp_folder, "video.mp4"),
        "audio": os.path.join(tmp_folder, "audio.mp3"),
        "srt": os.path.join(tmp_folder, "subtitles.srt"),
        "raw_transcript": os.path.join(tmp_folder, "raw_transcript.json"),
    }

    if any(url_like in media_path for url_like in ["http", "www"]):
        logging.info("Downloading media...")
        if save:
            download_video(media_path, output=STORE["video"])
            extract_audio(STORE["video"], STORE["audio"])
        else:
            download_audio(media_path, output=STORE["audio"])
    else:
        logging.info("Extracting audio...")
        STORE["video"] = media_path
        extract_audio(STORE["video"], STORE["audio"])

    logging.info("Transcribing audio...")
    diarize_config = None
    if diarize:
        diarize_config = yaml.load(open("secrets.yml"), Loader=yaml.FullLoader)
    transcription = whisperx_transcription(
        store=STORE,
        config=config["whisper"],
        diarize_config=diarize_config,
        model_id=model,
        language=language,
    )
    subtitles = segments_to_srt(segments=transcription["segments"])

    with open(STORE["srt"], "w") as f:
        f.writelines(subtitles)

    if save:
        output_file = f"{name}_subtitled.mp4"
        output_file = os.path.join(tmp_folder, output_file)
        write_video_with_subs(STORE["video"], STORE["srt"], output_file)
        logging.info(f"Subtitled video saved to {output_file}")

    return STORE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("media_path", help="Path to the media file")
    parser.add_argument(
        "-l",
        "--language",
        help="Language for translation",
        default="no",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model for transcription",
        default="NbAiLab/nb-whisper-large",
    )
    parser.add_argument(
        "-d", "--diarize", help="Whether to diarize the audio", action="store_true"
    )
    parser.add_argument(
        "-s", "--save", help="Save the subtitled video", action="store_true"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logging.info(f"Recieved args: {args}")

    store = transcribe(
        args.media_path,
        model=args.model,
        diarize=args.diarize,
        save=args.save,
        language=args.language,
    )
    logging.info(f"Transcription complete. See the output files in {store}")
