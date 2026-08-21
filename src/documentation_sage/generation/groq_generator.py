from groq import Groq

from documentation_sage.generation.base import BaseGenerator


class GroqGenerator(BaseGenerator):
    """
    Generator implementation using the official Groq SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "groq/compound",
        temperature: float = 0.2,
    ):

        self.client = Groq(
            api_key=api_key,
        )

        self.model = model
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        system_prompt = """
You are Documentation Sage, an AI assistant that answers questions
using the provided documentation.

Rules:

1. Answer ONLY using the provided context.
2. Do not use outside knowledge.
3. If the answer is not available in the context, say:
   "I couldn't find the answer in the provided documentation."
4. Be accurate and concise.
5. When possible, mention the source file used.
"""

        user_prompt = f"""
Context:

{context}


Question:

{query}


Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=self.temperature,
        )

        return response.choices[0].message.content or ""
