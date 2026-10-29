from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


SYSTEM_PROMPT = """你是一名严谨的课程学习助手。
请只根据用户提供的课程资料回答问题。
如果资料不足以回答，请明确说明“课程资料中没有足够信息”，不要编造。
回答使用中文，结构清晰，并尽量给出简短的理由或相关概念。"""


@dataclass(frozen=True)
class SourceSnippet:
    source: str
    content: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SourceSnippet]


def load_knowledge_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file does not exist: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Knowledge file is empty: {path}")

    return [Document(page_content=content, metadata={"source": path.name})]


def split_documents(documents: Sequence[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=60,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "],
    )
    return splitter.split_documents(list(documents))


def tokenize_for_bm25(text: str) -> list[str]:
    """Use character bigrams so Chinese text can be retrieved without a tokenizer."""
    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9_./+-]+", normalized)

    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))

    return tokens


class BM25DocumentRetriever:
    def __init__(self, documents: Sequence[Document], top_k: int = 4):
        if not documents:
            raise ValueError("Retriever requires at least one document.")

        self.documents = list(documents)
        self.top_k = top_k
        tokenized_documents = [
            tokenize_for_bm25(document.page_content)
            for document in self.documents
        ]
        self._index = BM25Okapi(tokenized_documents)

    def invoke(self, query: str) -> list[Document]:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )
        selected_indices = [
            index for index in ranked_indices if scores[index] > 0
        ][: self.top_k]
        return [self.documents[index] for index in selected_indices]


def build_retriever(path: Path, top_k: int = 4) -> BM25DocumentRetriever:
    documents = load_knowledge_documents(path)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No searchable chunks were created from the knowledge file.")

    return BM25DocumentRetriever(chunks, top_k=top_k)


def format_documents(documents: Sequence[Document]) -> str:
    if not documents:
        return "未检索到相关资料。"

    return "\n\n".join(
        f"[资料 {index}]\n{document.page_content.strip()}"
        for index, document in enumerate(documents, start=1)
    )


def create_answer_chain(
    api_key: str,
    base_url: str,
    model_name: str,
):
    if not api_key:
        raise ValueError("API Key 不能为空。")
    if not model_name:
        raise ValueError("模型名称不能为空。")

    model_kwargs: dict = {
        "api_key": api_key,
        "model": model_name,
        "temperature": 0.2,
        "timeout": 60,
        "max_retries": 2,
    }
    if base_url:
        model_kwargs["base_url"] = base_url

    model = ChatOpenAI(**model_kwargs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                """课程资料：
{context}

问题：
{question}""",
            ),
        ]
    )
    return prompt | model | StrOutputParser()


def answer_question(
    question: str,
    retriever: BM25DocumentRetriever,
    api_key: str,
    base_url: str,
    model_name: str,
) -> AnswerResult:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空。")

    documents = retriever.invoke(question)
    sources = [
        SourceSnippet(
            source=str(document.metadata.get("source", "knowledge.md")),
            content=document.page_content.strip(),
        )
        for document in documents
    ]

    if not documents:
        return AnswerResult(
            answer="课程资料中没有找到与该问题相关的内容。",
            sources=[],
        )

    chain = create_answer_chain(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )
    answer = chain.invoke(
        {
            "context": format_documents(documents),
            "question": question,
        }
    ).strip()

    if not answer:
        raise ValueError("模型返回了空内容。")

    return AnswerResult(answer=answer, sources=sources)
