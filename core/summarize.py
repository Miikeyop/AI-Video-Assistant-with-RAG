from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


model = ChatMistralAI(model="mistral-small-2603")


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def generate_title(transcript: str) -> str:
    title_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert title generator. Create a short, clear, professional title for the given transcript."
        ),
        (
            "human",
            """
Create one suitable title for this transcript.

Rules:
- Maximum 10 words
- No quotes
- Clear and professional
- Return only the title

Transcript:
{text}
"""
        )
    ])

    title_chain = title_template | model | StrOutputParser()

    title = title_chain.invoke({
        "text": transcript[:4000]
    })

    return title.strip()


def summarize(transcript: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of transcript concisely."),
        ("human", "{text}")
    ])

    chain = template | model | StrOutputParser()

    chunks = split_transcript(transcript)

    chunks_summary = []

    for i in chunks:
        summary = chain.invoke({"text": i})
        chunks_summary.append(summary)

    combined = "\n\n".join(chunks_summary)

    final_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries into one professional summary."
        ),
        ("human", "{text}")
    ])

    chain_final = final_template | model | StrOutputParser()

    final_combined = chain_final.invoke({
        "text": combined
    })

    return final_combined