import os
from pathlib import Path
from typing import Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


VECTORSTORE_DIR = Path("vectorstore")
INDEX_NAME = "faiss_index"


PROMPT = """You are a helpful assistant.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
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

    # embeddings = OllamaEmbeddings(model="llama3.1")
    
    EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(
        model="llama3.1",
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT)

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}"
            for d in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain, retriever