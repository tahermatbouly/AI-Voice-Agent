import os
from groq import Groq

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Say hello in Egyptian Arabic."
        }
    ],
)

print(response.choices[0].message.content)