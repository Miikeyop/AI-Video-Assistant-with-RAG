import whisper
import os
import time




WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully")

    return _model


def transcribe_text(chunk_path: str, translate: bool = True) -> str:
    model = load_model()

    task = "translate" if translate else "transcribe"

    start_time = time.perf_counter()
    print(f"Started: {chunk_path}")

    result = model.transcribe(
        chunk_path,
        task=task,
        fp16=False,
        verbose=False,
        temperature=0,
        condition_on_previous_text=False,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Finished chunk in {elapsed:.1f} seconds")

    return result["text"]


def transcribe_all(chunks: list, translate: bool = True) -> str:
    final_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"\nTranscribing chunk {i + 1}/{len(chunks)}\n")

        try:
            text = transcribe_text(chunk, translate=translate)
            final_transcript += text + " "

        except Exception as e:
            print(f"Chunk {i + 1} failed: {e}")
            final_transcript += f"\n[Chunk {i + 1} failed]\n"

    print("\nTranscription done\n")

    return final_transcript