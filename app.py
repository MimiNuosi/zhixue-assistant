from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import AnswerResult, answer_question, build_retriever, split_documents, load_knowledge_documents


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "knowledge.md"

load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="智学助手",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 920px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    [data-testid="stSidebar"] {
        background: #F1F4EF;
        border-right: 1px solid #D9E0D8;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 8px;
    }
    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_retriever(knowledge_file_signature: int):
    del knowledge_file_signature
    return build_retriever(DATA_PATH, top_k=4)


@st.cache_data(show_spinner=False)
def get_chunk_count(knowledge_file_signature: int) -> int:
    del knowledge_file_signature
    documents = load_knowledge_documents(DATA_PATH)
    return len(split_documents(documents))


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("引用片段"):
        for index, source in enumerate(sources, start=1):
            st.markdown(f"**片段 {index} · {source['source']}**")
            st.text(source["content"])


if not DATA_PATH.exists():
    st.error(f"未找到知识库文件：{DATA_PATH}")
    st.stop()

file_signature = DATA_PATH.stat().st_mtime_ns
retriever = get_retriever(file_signature)
chunk_count = get_chunk_count(file_signature)

st.title("智学助手")
st.caption(f"基于 LangChain 的课程资料问答 · {DATA_PATH.name} · {chunk_count} 个片段")

with st.sidebar:
    st.subheader("模型配置")
    api_key = st.text_input(
        "API Key",
        value=os.getenv("LLM_API_KEY", ""),
        type="password",
        help="在线模型的 API Key。只保存在当前会话或本机 .env 中。",
    ).strip()
    base_url = st.text_input(
        "Base URL",
        value=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        help="填写服务商提供的 OpenAI 兼容接口地址。",
    ).strip()
    model_name = st.text_input(
        "模型名称",
        value=os.getenv("LLM_MODEL", "deepseek-chat"),
    ).strip()

    if api_key and model_name:
        st.caption("模型配置已就绪")
    else:
        st.caption("请填写 API Key 和模型名称")

    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

placeholder = "输入课程问题" if api_key else "请先在侧边栏填写 API Key"
question = st.chat_input(placeholder, disabled=not api_key)

if question:
    question = question.strip()
    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("正在检索课程资料并生成回答"):
                    result: AnswerResult = answer_question(
                        question=question,
                        retriever=retriever,
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                    )
                st.markdown(result.answer)
                source_payload = [
                    {"source": item.source, "content": item.content}
                    for item in result.sources
                ]
                render_sources(source_payload)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": source_payload,
                    }
                )
            except Exception as exc:
                error_message = f"模型调用失败：{exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
