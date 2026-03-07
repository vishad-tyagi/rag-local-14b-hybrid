import math
import os
from typing import Iterable, List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _require_token() -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found. Put it in your .env file.")


def _to_python_list(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _mean_pool(matrix: List[List[float]]) -> List[float]:
    if not matrix:
        return []
    cols = len(matrix[0])
    sums = [0.0] * cols
    for row in matrix:
        for i, val in enumerate(row):
            sums[i] += float(val)
    return [x / len(matrix) for x in sums]


def _coerce_to_vector(output) -> List[float]:
    """
    HF feature_extraction can return either:
    - a 1D vector
    - a 2D matrix
    Convert both into one 1D vector.
    """
    output = _to_python_list(output)

    if not output:
        return []

    if isinstance(output[0], (int, float)):
        return [float(x) for x in output]

    if isinstance(output[0], list):
        return _mean_pool(output)

    raise ValueError(f"Unexpected embedding output shape/type: {type(output)}")


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class HFEmbeddingsAPI(Embeddings):
    def __init__(self, model: str = EMBEDDING_MODEL):
        _require_token()
        self.model = model
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN,
        )

    def embed_query(self, text: str) -> List[float]:
        result = self.client.feature_extraction(
            text,
            model=self.model,
        )
        vector = _coerce_to_vector(result)
        return _l2_normalize(vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


class HFChatLLM:
    """
    Uses Hugging Face CHAT COMPLETIONS.
    Important:
    - Do NOT force provider="hf-inference" here.
    - Let Hugging Face route to the available provider for this model.
    """

    def __init__(self, model: str = CHAT_MODEL):
        _require_token()
        self.model = model
        self.client = InferenceClient(
            api_key=HF_TOKEN,
        )

    def answer(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        return completion.choices[0].message.content or ""

    def stream_answer(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 700,
        temperature: float = 0.2,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError, KeyError):
                delta = None

            if delta:
                yield delta
                
                

# class HFChatLLM:
#     """
#     Uses Hugging Face TEXT GENERATION instead of chat completions.
#     This fixes the 404 / conversational-task mismatch for the chosen model.
#     """

#     def __init__(self, model: str = CHAT_MODEL):
#         _require_token()
#         self.model = model
#         self.client = InferenceClient(
#             # provider="hf-inference",
#             api_key=HF_TOKEN,
#         )

#     def _build_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
#         if system_prompt:
#             return f"""<|system|>
# {system_prompt}
# <|user|>
# {prompt}
# <|assistant|>
# """
#         return f"""<|user|>
# {prompt}
# <|assistant|>
# """

#     def answer(
#         self,
#         prompt: str,
#         system_prompt: str | None = None,
#         max_tokens: int = 700,
#         temperature: float = 0.2,
#     ) -> str:
#         full_prompt = self._build_prompt(prompt, system_prompt)

#         output = self.client.text_generation(
#             prompt=full_prompt,
#             model=self.model,
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             return_full_text=False,
#             do_sample=True,
#         )

#         return output.strip()

#     def stream_answer(
#         self,
#         prompt: str,
#         system_prompt: str | None = None,
#         max_tokens: int = 700,
#         temperature: float = 0.2,
#     ) -> Iterable[str]:
#         full_prompt = self._build_prompt(prompt, system_prompt)

#         stream = self.client.text_generation(
#             prompt=full_prompt,
#             model=self.model,
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             return_full_text=False,
#             do_sample=True,
#             stream=True,
#         )

#         for chunk in stream:
#             if chunk:
#                 yield chunk



