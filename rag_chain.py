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
from typing import List, Tuple

from langchain_community.vectorstores import FAISS

from hf_models import LocalEmbeddings, MLXLocalLLM

VECTORSTORE_DIR = Path("vectorstore")
INDEX_NAME = "faiss_index"

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = MLXLocalLLM()

    class RAGChain:
        def invoke(self, question: str) -> str:
            docs = retriever.invoke(question)
            prompt = build_prompt(question, docs)
            # 7B models benefit from a slightly higher max_token count for synthesis
            return llm.answer(prompt=prompt, system_prompt=SYSTEM_PROMPT, max_tokens=800)

    return RAGChain(), retriever, llm