import os

from dotenv import load_dotenv

from documentation_sage.generation.groq_generator import (
    GroqGenerator,
)

load_dotenv()


generator = GroqGenerator(
    api_key=os.environ["GROQ_API_KEY"],
)


query = "How do I handle exceptions in Python?"

context = """
[Source: errors.txt]

Python handles exceptions using try and except statements.

Example:

try:
    value = int("hello")
except ValueError:
    print("Invalid value")
"""


answer = generator.generate(
    query=query,
    context=context,
)


print("\nQuestion:")
print(query)

print("\nAnswer:")
print(answer)
