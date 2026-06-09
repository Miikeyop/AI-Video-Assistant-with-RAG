from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


PERSIST_DIR = "chroma-db"

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


def create_vector_store(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    text_chunks = splitter.split_text(transcript)

    documents = []

    for i, chunk in enumerate(text_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": "meeting_transcript",
                "chunk_id": i + 1
            }
        )

        documents.append(doc)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )

    return vector_store


def load_vector_store():
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding
    )

    return vector_store


def retrieve_context(question: str, k: int = 4) -> str:
    vector_store = load_vector_store()

    docs = vector_store.similarity_search(
        question,
        k=k
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    return context