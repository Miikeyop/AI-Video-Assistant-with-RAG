import os
import streamlit as st
from dotenv import load_dotenv

from main import run_pipeline

load_dotenv()


st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide"
)


def save_uploaded_file(uploaded_file):
    os.makedirs("uploaded_files", exist_ok=True)

    file_path = os.path.join("uploaded_files", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: #ffffff;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 30px;
    }

    .card {
        padding: 22px;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        margin-bottom: 18px;
        color: #111827;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }

    .card h3 {
        color: #111827 !important;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }

    .metric-card {
        padding: 24px;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        text-align: center;
        min-height: 170px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }

    .metric-card h3 {
        color: #111827 !important;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .metric-card p {
        color: #374151 !important;
        font-size: 16px;
        font-weight: 500;
        line-height: 1.6;
        margin: 0;
    }

    .section-heading {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<div class='main-title'>🎥 AI Video / Meeting Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Transcribe, summarize, extract insights, and chat with your video or meeting transcript.</div>",
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("📥 Input")

    input_type = st.radio(
        "Choose input type",
        ["YouTube URL", "Upload File"]
    )

    source = None

    if input_type == "YouTube URL":
        source = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload audio or video file",
            type=["mp3", "wav", "m4a", "mp4", "webm", "aac"]
        )

        if uploaded_file is not None:
            source = save_uploaded_file(uploaded_file)
            st.success("File uploaded successfully")

    st.divider()

    start_button = st.button(
        "🚀 Start Processing",
        use_container_width=True
    )

    st.caption("Output transcript will be generated in English.")


if start_button:
    if not source:
        st.error("Please enter a YouTube URL or upload a file.")

    else:
        with st.spinner("Processing your video/audio... This may take some time."):
            try:
                result = run_pipeline(source, translate=True)

                st.session_state.result = result
                st.session_state.chat_history = []

                st.success("Processing completed successfully!")

            except Exception as e:
                st.error("Something went wrong.")
                st.exception(e)


if st.session_state.result is None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class='metric-card'>
                <h3>🎧 Audio Processing</h3>
                <p>Download or upload audio/video and convert it for transcription.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class='metric-card'>
                <h3>📝 AI Summary</h3>
                <p>Generate title, summary, action items, decisions, and questions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class='metric-card'>
                <h3>💬 RAG Chat</h3>
                <p>Ask questions from your transcript using vector search.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info("Start by giving a YouTube URL or uploading a file from the sidebar.")


else:
    result = st.session_state.result

    st.markdown("## 📌 Generated Title")

    st.markdown(
        f"""
        <div class='card'>
            <h3>{result["title"]}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Summary",
            "✅ Action Items",
            "🔑 Key Decisions",
            "❓ Questions",
            "📝 Transcript"
        ]
    )

    with tab1:
        st.markdown(
            "<div class='section-heading'>Summary</div>",
            unsafe_allow_html=True
        )
        st.write(result["summary"])

    with tab2:
        st.markdown(
            "<div class='section-heading'>Action Items</div>",
            unsafe_allow_html=True
        )
        st.write(result["action_items"])

    with tab3:
        st.markdown(
            "<div class='section-heading'>Key Decisions</div>",
            unsafe_allow_html=True
        )
        st.write(result["key_decisions"])

    with tab4:
        st.markdown(
            "<div class='section-heading'>Open Questions</div>",
            unsafe_allow_html=True
        )
        st.write(result["open_questions"])

    with tab5:
        st.markdown(
            "<div class='section-heading'>Full Transcript</div>",
            unsafe_allow_html=True
        )

        st.text_area(
            "Transcript",
            result["transcript"],
            height=450
        )

    st.divider()

    st.markdown("## 💬 Chat with Your Video / Meeting")

    rag_chain = result["rag_chain"]

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

    question = st.chat_input("Ask anything from the transcript...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Finding answer from transcript..."):
            answer = rag_chain.invoke(question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )