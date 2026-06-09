import yt_dlp
import os
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": False,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = filename.replace(".webm", ".wav").replace(".m4a", ".wav")

    return filename


def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]

        chunk_path = os.path.splitext(wav_path)[0] + f"_chunk_{i + 1}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

        duration_seconds = len(chunk) / 1000
        print(f"Created chunk {i + 1}: {duration_seconds:.1f} seconds")

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("\nYouTube URL: Downloading audio...\n")

        downloaded_path = download_youtube_audio(source)

        print("\nConverting audio to 16 kHz mono WAV...\n")

        wav_path = convert_to_wav(downloaded_path)

    else:
        print("\nLocal File: Converting to WAV...\n")

        wav_path = convert_to_wav(source)

    print("\nChunking audio\n")

    chunks = chunk_audio(wav_path)

    print("\nAudio Ready\n")

    return chunks