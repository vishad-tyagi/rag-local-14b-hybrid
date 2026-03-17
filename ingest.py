# from pathlib import Path
# from typing import List

# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from hf_models import LocalEmbeddings

# DATA_DIR = Path("data")
# VECTORSTORE_DIR = Path("vectorstore")
# INDEX_NAME = "faiss_index"


# def load_documents(data_dir: Path) -> List:
#     docs = []

#     for path in data_dir.rglob("*"):
#         if path.is_dir():
#             continue

#         suffix = path.suffix.lower()

#         if suffix == ".pdf":
#             loader = PyPDFLoader(str(path))
#             docs.extend(loader.load())

#         elif suffix in {".txt", ".md"}:
#             loader = TextLoader(str(path), encoding="utf-8")
#             docs.extend(loader.load())

#     return docs


# def main() -> None:
#         if not DATA_DIR.exists():
#             raise FileNotFoundError(f"Missing data directory: {DATA_DIR.resolve()}")

#         docs = load_documents(DATA_DIR)
#         if not docs:
#             raise ValueError(f"No supported files found in {DATA_DIR.resolve()}")

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=900,
#             chunk_overlap=150,
#             separators=["\n\n", "\n", ". ", " ", ""],
#         )
#         chunks = splitter.split_documents(docs)

#         embeddings = LocalEmbeddings()

#         vectorstore = FAISS.from_documents(chunks, embeddings)
#         VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
#         vectorstore.save_local(str(VECTORSTORE_DIR / INDEX_NAME))

#         print(f"Indexed {len(docs)} documents into {len(chunks)} chunks.")
#         print(f"Saved FAISS index to: {VECTORSTORE_DIR / INDEX_NAME}")


# if __name__ == "__main__":
#     main()



from pathlib import Path
from typing import List

# CHANGED: Using PyMuPDFLoader for better layout preservation in 2026
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hf_models import LocalEmbeddings

DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
INDEX_NAME = "faiss_index"

def load_documents(data_dir: Path) -> List:
    docs = []

    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # CHANGED: PyMuPDFLoader is faster and cleaner than PyPDF
            loader = PyMuPDFLoader(str(path))
            docs.extend(loader.load())

        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())

    return docs

def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR.resolve()}")

    docs = load_documents(DATA_DIR)
    if not docs:
        raise ValueError(f"No supported files found in {DATA_DIR.resolve()}")

    # CHANGED: Optimized chunk_size and overlap for the 7B model's higher intelligence
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # Increased for more context per retrieval
        chunk_overlap=200,    # Ensures continuity across boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = LocalEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR / INDEX_NAME))

    print(f"Indexed {len(docs)} documents into {len(chunks)} chunks.")
    print(f"Saved FAISS index to: {VECTORSTORE_DIR / INDEX_NAME}")

if __name__ == "__main__":
    main()