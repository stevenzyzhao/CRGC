import json
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    MODELS, VLLM_BASE_URL, TEMPERATURE
)


class LLMClient:
    def __init__(self):
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        if ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        if GOOGLE_API_KEY:
            self.google_client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            self.google_client = None
        self.vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
    def generate(self, model_name, prompt, system_prompt="", temperature=None, n=1):
        if temperature is None:
            temperature = TEMPERATURE
        model_config = MODELS[model_name]
        provider = model_config["provider"]
        model_id = model_config["model_id"]

        if provider == "openai":
            return self._generate_openai(model_id, prompt, system_prompt, temperature, n)
        elif provider == "anthropic":
            return self._generate_anthropic(model_id, prompt, system_prompt, temperature, n)
        elif provider == "google":
            return self._generate_google(model_id, prompt, system_prompt, temperature, n)
        elif provider == "vllm":
            return self._generate_vllm(model_id, prompt, system_prompt, temperature, n)

    def _generate_openai(self, model_id, prompt, system_prompt, temperature, n):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            n=n,
        )
        if n == 1:
            return response.choices[0].message.content
        return [choice.message.content for choice in response.choices]

    def _generate_anthropic(self, model_id, prompt, system_prompt, temperature, n):
        results = []
        for _ in range(n):
            kwargs = {"model": model_id, "max_tokens": 4096, "temperature": temperature,
                      "messages": [{"role": "user", "content": prompt}]}
            if system_prompt:
                kwargs["system"] = system_prompt
            response = self.anthropic_client.messages.create(**kwargs)
            results.append(response.content[0].text)
        if n == 1:
            return results[0]
        return results

    def _generate_google(self, model_id, prompt, system_prompt, temperature, n):
        results = []
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt or None,
        )
        for _ in range(n):
            response = self.google_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )
            results.append(response.text)
        if n == 1:
            return results[0]
        return results

    def _generate_vllm(self, model_id, prompt, system_prompt, temperature, n):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.vllm_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            n=n,
        )
        if n == 1:
            return response.choices[0].message.content
        return [choice.message.content for choice in response.choices]

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
    def generate_json(self, model_name, prompt, system_prompt=""):
        response = self.generate(model_name, prompt, system_prompt, temperature=0.0)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        return json.loads(response.strip())


llm_client = LLMClient()
