
import os
import httpx

async def get_chatbot_logic_response(user_input: str, context: str = "") -> str:
    """
    Asynchronously call the chatbot API (e.g., OpenRouter) using the provided user_input
    and optional context, and return the chatbot's text response.
    """
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": "You are an advanced AI assistant."},
            
            {"role": "user", "content": f"Context: {context}\n\n{user_input}"}
        ],
        "temperature": 0.75,
        "max_tokens": 500
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raises an exception for HTTP errors
        data = response.json()
        # Extract and return the chatbot's reply from the API response
        return data["choices"][0]["message"]["content"]
