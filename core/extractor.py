from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatMistralAI(model="mistral-small-2603")
parser = StrOutputParser()


def extract_action_items(transcript: str) -> str:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all action items. For each provide:\n"
            "- Task description\n"
            "- Owner (who is responsible)\n"
            "- Deadline (if mentioned, else write 'Not specified')\n\n"
            "Format as a numbered list. If none found say 'No action items found.'"
        ),
        (
            "human",
            "Meeting transcript:\n\n{transcript}"
        )
    ])

    chain = template | model | parser

    return chain.invoke({
        "transcript": transcript
    })


def extract_decisions(transcript: str) -> str:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all important decisions that were made.\n\n"
            "For each decision provide:\n"
            "- Decision description\n"
            "- Reason/context if mentioned\n"
            "- Who made or agreed to the decision if mentioned\n\n"
            "Format as a numbered list. If none found say 'No decisions found.'"
        ),
        (
            "human",
            "Meeting transcript:\n\n{transcript}"
        )
    ])

    chain = template | model | parser

    return chain.invoke({
        "transcript": transcript
    })


def extract_questions(transcript: str) -> str:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all important questions asked or questions that remain unanswered.\n\n"
            "For each question provide:\n"
            "- Question\n"
            "- Asked by if mentioned\n"
            "- Answer/status if mentioned, else write 'Not answered'\n\n"
            "Format as a numbered list. If none found say 'No questions found.'"
        ),
        (
            "human",
            "Meeting transcript:\n\n{transcript}"
        )
    ])

    chain = template | model | parser

    return chain.invoke({
        "transcript": transcript
    })


def extract_all(transcript: str) -> dict:
    action_items = extract_action_items(transcript)
    decisions = extract_decisions(transcript)
    questions = extract_questions(transcript)

    return {
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions
    }