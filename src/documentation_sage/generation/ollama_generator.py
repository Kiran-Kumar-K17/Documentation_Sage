import requests

from documentation_sage.generation.base import BaseGenerator


class OllamaGenerator(BaseGenerator):

    def __init__(
        self,
        model: str = "phi4-mini:3.8b",
        temperature: float = 0.2,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        system_prompt = """
You are Documentation Sage, a documentation-based AI assistant.

Rules:
1. Answer ONLY using the provided context.
2. Do not use outside knowledge.
3. Do not add information that is not supported by the context.
4. If the answer is not available in the context, say exactly:
   "I couldn't find the answer in the provided documentation."
5. Be accurate and concise.
6. Do not explain your reasoning.
7. Do not include a Sources section.

Return only the final answer.
"""

        user_prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self.temperature,
                },
            },
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]
