from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS

from hf_models import HFEmbeddingsAPI, HFChatLLM

VECTORSTORE_DIR = Path("vectorstore")
INDEX_NAME = "faiss_index"

SYSTEM_PROMPT = (
    "You are Butler, a helpful assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say: "
    "\"I don't know based on the provided documents.\""
)


def build_prompt(question: str, docs: List) -> str:
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

    return f"""Context:
{context}

Question:
{question}
"""


def build_rag() -> Tuple:
    index_path = VECTORSTORE_DIR / INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path.resolve()}. Run: python ingest.py"
        )

    embeddings = HFEmbeddingsAPI()
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = HFChatLLM()

    class RAGChain:
        def invoke(self, question: str) -> str:
            docs = retriever.invoke(question)
            prompt = build_prompt(question, docs)
            return llm.answer(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    return RAGChain(), retriever, llm