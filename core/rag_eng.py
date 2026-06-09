import os
from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import create_vector_store, load_vector_store, retrieve_context


def get_llm():
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


def get_rag_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert meeting/video assistant.

Answer the user's question based ONLY on the transcript context given below.

If the answer is not available in the context, say:
"I could not find this information in the transcript."

Keep the answer clear, simple, and concise.

Transcript Context:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ])


def build_rag_chain(transcript: str):
    create_vector_store(transcript)

    llm = get_llm()

    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": RunnableLambda(lambda question: retrieve_context(question, k=4)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    load_vector_store()

    llm = get_llm()

    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": RunnableLambda(lambda question: retrieve_context(question, k=4)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"\nQuestion: {question}\n")

    answer = rag_chain.invoke(question)

    print(f"\nAnswer: {answer}\n")

    return answer