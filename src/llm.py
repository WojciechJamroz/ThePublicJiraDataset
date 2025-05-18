import logging
from typing import Optional
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

# Gemini
import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel

# Local LLM (transformers)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class LLMWrapper:
    def __init__(self, backend: str = "gemini", local_model_name: Optional[str] = None, device: Optional[str] = None):
        self.backend = backend
        self.local_model_name = local_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        if backend == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in environment")
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        elif backend == "local":
            if not local_model_name:
                raise ValueError("local_model_name must be provided for local backend")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(local_model_name, torch_dtype="auto", device_map="auto")
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device=="cuda" else -1)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate(self, prompt: str, max_new_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        if self.backend == "gemini":
            resp = self.model.generate_content(prompt)
            return resp.text
        elif self.backend == "local":
            # Use chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                chat = []
                if system_prompt:
                    chat.append({"role": "system", "content": system_prompt})
                chat.append({"role": "user", "content": prompt})
                prompt_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = prompt
            # Use model.generate for chat models, otherwise pipeline fallback
            if hasattr(self.model, "generate") and hasattr(self.tokenizer, "__call__"):
                input_ids = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                output = self.model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
                decoded = self.tokenizer.decode(output[0][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
                return decoded.strip()
            else:
                outputs = self.pipeline(prompt_text, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
                return outputs[0]["generated_text"][len(prompt_text):].strip()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
