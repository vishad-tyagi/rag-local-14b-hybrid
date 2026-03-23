from typing import Iterable, List, Optional

import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from mlx_lm import load, generate, stream_generate
from pydantic.v1 import PrivateAttr
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
# CHAT_MODEL = "mlx-community/defog-llama-3-sqlcoder-8b" 

_EMBED_MODEL = None
_EMBED_DEVICE = None
_MLX_MODEL = None
_MLX_TOKENIZER = None


def _embedding_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        global _EMBED_MODEL, _EMBED_DEVICE
        self.model_name = model_name

        if _EMBED_MODEL is None:
            _EMBED_DEVICE = _embedding_device()
            _EMBED_MODEL = SentenceTransformer(self.model_name, device=_EMBED_DEVICE)

        self.device = _EMBED_DEVICE
        self.model = _EMBED_MODEL

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class MLXLocalLLM:
    def __init__(self, model_name: str = CHAT_MODEL):
        global _MLX_MODEL, _MLX_TOKENIZER
        self.model_name = model_name

        if _MLX_MODEL is None or _MLX_TOKENIZER is None:
            _MLX_MODEL, _MLX_TOKENIZER = load(self.model_name)

        self.model = _MLX_MODEL
        self.tokenizer = _MLX_TOKENIZER

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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
        system_prompt: Optional[str] = None,
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
        # FIX: Strip Qwen special tokens before returning
        output = output.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        return output

    def stream_answer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 700,
    ) -> Iterable[str]:
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        for response in stream_generate(
            self.model,
            self.tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
        ):
            # CHANGED: Safely handle the MLX GenerationResponse object
            # If it has a text attribute, use it (even if empty). 
            # This prevents the final metadata chunk from being converted to a string.
            if hasattr(response, "text"):
                text = response.text
            elif isinstance(response, str):
                text = response
            else:
                text = ""
                
            # FIX: Prevent special tokens from streaming to the frontend
            if "<|im_end|>" in text or "<|endoftext|>" in text:
                text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            
            if text:
                yield text

class LangChainMLXLLM(LLM):
    model_name: str = CHAT_MODEL
    
    # FIXED: The proper Pydantic way to handle unvalidated private attributes
    _engine: MLXLocalLLM = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "mlx_local"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

    def __init__(self, model_name: str = CHAT_MODEL, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        # Now you can assign this normally without Pydantic throwing an error
        self._engine = MLXLocalLLM(model_name=model_name)

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        text = self._engine.answer(
            prompt=prompt,
            system_prompt=None,
            max_tokens=kwargs.get("max_tokens", 256),
        )

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]

        return text