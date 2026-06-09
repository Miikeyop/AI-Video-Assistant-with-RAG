from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_decisions, extract_questions
from core.rag_eng import build_rag_chain, ask_question


def run_pipeline(source: str, translate: bool = True) -> dict:
    print("\nStarting AI Video Assistant...\n")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, translate=translate)

    print(f"\nRaw transcription first 300 characters:\n{transcript[:300]}\n")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_items = extract_action_items(transcript)

    decisions = extract_decisions(transcript)

    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()

    translate_input = input("Translate transcript to English? yes/no: ").strip().lower()

    translate = True

    if translate_input in ["no", "n", "false"]:
        translate = False

    result = run_pipeline(source, translate=translate)

    print("\n" + "=" * 60)

    print(f"\nTITLE:\n{result['title']}")

    print(f"\nSUMMARY:\n{result['summary']}")

    print(f"\nACTION ITEMS:\n{result['action_items']}")

    print(f"\nKEY DECISIONS:\n{result['key_decisions']}")

    print(f"\nOPEN QUESTIONS:\n{result['open_questions']}")

    print("\n" + "=" * 60)

    print("\nChat with your meeting/video transcript")
    print("Type 'exit' to quit\n")

    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\nAssistant: {answer}\n")