import requests
from settings import get_settings

def call_openrouter(prompt: str):
    settings = get_settings().llm
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": settings.system_prompt_arabic},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, json=data, headers=headers, timeout=settings.request_timeout_seconds)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]