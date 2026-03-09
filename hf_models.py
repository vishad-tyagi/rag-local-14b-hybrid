from typing import Iterable, List

import torch
from langchain_core.embeddings import Embeddings
from mlx_lm import load, generate, stream_generate
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def _embedding_device() -> str:
    """
    Use MPS on Apple Silicon if available, otherwise CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.device = _embedding_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class MLXLocalLLM:
    def __init__(self, model_name: str = CHAT_MODEL):
        self.model_name = model_name

        # On first run this downloads the MLX model to local HF cache.
        # Later runs use the cached local copy.
        self.model, self.tokenizer = load(self.model_name)

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def answer(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 700,
    ) -> str:
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        output = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        return output.strip()

    def stream_answer(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 700,
    ) -> Iterable[str]:
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        for response in stream_generate(
            self.model,
            self.tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
        ):
            if getattr(response, "text", None):
                yield response.text