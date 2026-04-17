# from pathlib import Path
# from typing import List, Tuple

# from langchain_community.vectorstores import FAISS

# from hf_models import LocalEmbeddings, MLXLocalLLM

# VECTORSTORE_DIR = Path("vectorstore")
# INDEX_NAME = "faiss_index"

# SYSTEM_PROMPT = (
#     "You are Butler, a helpful assistant. "
#     "Answer the user's question using ONLY the provided context. "
#     "If the answer is not in the context, say: "
#     "\"I don't know based on the provided documents.\""
# )


# def build_prompt(question: str, docs: List) -> str:
#     context = "\n\n".join(
#         f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
#         for d in docs
#     )

#     return f"""Context:
# {context}

# Question:
# {question}
# """


# def build_rag() -> Tuple:
#     index_path = VECTORSTORE_DIR / INDEX_NAME
#     if not index_path.exists():
#         raise FileNotFoundError(
#             f"FAISS index not found at {index_path.resolve()}. Run: python ingest.py"
#         )

#     embeddings = LocalEmbeddings()

#     vectorstore = FAISS.load_local(
#         str(index_path),
#         embeddings,
#         allow_dangerous_deserialization=True,
#     )

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
#     llm = MLXLocalLLM()

#     class RAGChain:
#         def invoke(self, question: str) -> str:
#             docs = retriever.invoke(question)
#             prompt = build_prompt(question, docs)
#             return llm.answer(prompt=prompt, system_prompt=SYSTEM_PROMPT)

#     return RAGChain(), retriever, llm



from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from hf_models import LocalEmbeddings, MLXLocalLLM

VECTORSTORE_DIR = Path("vectorstore")
INDEX_NAME = "faiss_index"
VECTOR_K = 5
BM25_K = 5
FINAL_K = 5
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RRF_K = 60

# UPDATED: Refined for the 7B model to allow for better synthesis 
# while maintaining strict grounding.
SYSTEM_PROMPT = (
    "You are Butler, a sophisticated local assistant. "
    "Your task is to answer the user's question using ONLY the provided context. "
    "Guidelines:\n"
    "1. If the answer is not in the context, say exactly: 'I don't know based on the provided documents.'\n"
    "2. Be concise but thorough.\n"
    "3. Do not use outside knowledge or hallucinate details not present in the sources.\n"
    "4. If multiple sources conflict, mention the discrepancy."
)


def build_prompt(question: str, docs: List) -> str:
    # UPDATED: Added structural markers to help the 7B model 
    # distinguish between multiple retrieved chunks.
    context_blocks = []
    for i, d in enumerate(docs):
        source = d.metadata.get('source', 'unknown')
        context_blocks.append(f"--- SOURCE {i+1} [{source}] ---\n{d.page_content}")
    
    context = "\n\n".join(context_blocks)

    return f"""Use the following pieces of retrieved context to answer the question. 

CONTEXT:
{context}

QUESTION:
{question}

HELPFUL ANSWER:"""


def _doc_key(doc) -> Tuple[str, str]:
    return (
        doc.metadata.get("source", "unknown"),
        doc.page_content,
    )


def _load_documents_from_vectorstore(vectorstore: FAISS) -> List:
    docstore = getattr(vectorstore, "docstore", None)
    docs_by_id = getattr(docstore, "_dict", {}) if docstore else {}
    return list(docs_by_id.values())


class HybridRetriever:
    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        final_k: int = FINAL_K,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.final_k = final_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def _add_ranked_scores(
        self,
        scores: Dict[Tuple[str, str], float],
        docs_by_key: Dict[Tuple[str, str], object],
        docs: List,
        weight: float,
    ) -> None:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            docs_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)

    def invoke(self, question: str) -> List:
        vector_docs = self.vector_retriever.invoke(question)
        bm25_docs = self.bm25_retriever.invoke(question)

        scores = {}
        docs_by_key = {}
        self._add_ranked_scores(scores, docs_by_key, vector_docs, self.vector_weight)
        self._add_ranked_scores(scores, docs_by_key, bm25_docs, self.bm25_weight)

        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [docs_by_key[key] for key in ranked_keys[: self.final_k]]


def build_rag() -> Tuple:
    index_path = VECTORSTORE_DIR / INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path.resolve()}. Run: python ingest.py"
        )

    embeddings = LocalEmbeddings()

    # Allow dangerous deserialization is required for loading local FAISS
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Increased k to 5 to take advantage of the 7B model's larger context window
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})

    documents = _load_documents_from_vectorstore(vectorstore)
    if not documents:
        raise ValueError("No documents found in FAISS docstore. Run: python ingest.py")

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = BM25_K

    retriever = HybridRetriever(vector_retriever, bm25_retriever)
    llm = MLXLocalLLM()

    class RAGChain:
        def invoke(self, question: str) -> str:
            docs = retriever.invoke(question)
            prompt = build_prompt(question, docs)
            # 7B models benefit from a slightly higher max_token count for synthesis
            return llm.answer(prompt=prompt, system_prompt=SYSTEM_PROMPT, max_tokens=800)

    return RAGChain(), retriever, llm
