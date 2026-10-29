from pathlib import Path

from rag import (
    build_retriever,
    format_documents,
    load_knowledge_documents,
    split_documents,
    tokenize_for_bm25,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge.md"


def test_knowledge_file_is_split_into_chunks():
    documents = load_knowledge_documents(DATA_PATH)
    chunks = split_documents(documents)

    assert documents
    assert len(chunks) >= 5
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["source"] == "knowledge.md" for chunk in chunks)


def test_bm25_finds_langchain_content():
    retriever = build_retriever(DATA_PATH, top_k=3)
    results = retriever.invoke("LangChain 是什么")

    assert results
    assert any("LangChain" in document.page_content for document in results)


def test_bm25_finds_rag_content():
    retriever = build_retriever(DATA_PATH, top_k=3)
    results = retriever.invoke("RAG 的处理流程是什么")

    assert results
    assert any("检索增强生成" in document.page_content for document in results)


def test_chinese_bigram_tokenizer_keeps_keyword_parts():
    tokens = tokenize_for_bm25("智能应用开发与 LangChain")

    assert "智能" in tokens
    assert "langchain" in tokens


def test_format_documents_adds_context_labels():
    documents = load_knowledge_documents(DATA_PATH)
    formatted = format_documents(documents)

    assert formatted.startswith("[资料 1]")
    assert "LangChain" in formatted
