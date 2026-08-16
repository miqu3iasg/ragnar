# ref:
# https://openrouter.ai/docs/api_reference/overview
# repository with code examples: https://github.com/openai/openai-python/
from openai import OpenAI

# You can import your env variables here simply declaring them whithout much effort in `config.py`
from infrastructure.llm.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def get_completion(content: str):
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
            "X-OpenRouter-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
        },
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    # We don't need of the whole object, we only need the generated response
    return completion.choices[0].message.content
