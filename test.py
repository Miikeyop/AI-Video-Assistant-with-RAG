from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_all


source = "https://www.youtube.com/watch?v=IVGjBxqygmI"


audio_chunks = process_input(source)


transcript = transcribe_all(audio_chunks)


title = generate_title(transcript)


summary = summarize(transcript)


extracted_data = extract_all(transcript)


print("\n================ TITLE ================\n")
print(title)

print("\n================ TRANSCRIPT ================\n")
print(transcript)

print("\n================ SUMMARY ================\n")
print(summary)

print("\n================ ACTION ITEMS ================\n")
print(extracted_data["action_items"])

print("\n================ DECISIONS ================\n")
print(extracted_data["decisions"])

print("\n================ QUESTIONS ================\n")
print(extracted_data["questions"])