# llama_recommender.py
import os
import requests

GROQ_API_KEY = "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"
LLM_MODEL = "llama3-8b-8192"

def get_suggestion(disease_name):
    prompt = f"""
You are a dermatologist AI assistant. For the skin disease '{disease_name}', suggest:
1. Ingredients to avoid
2. Skin type most affected
3. Safe ingredients
4. Product recommendations (general)
5. Additional skincare advice
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI dermatologist."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]
