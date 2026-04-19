from typing import Optional

SYSTEM_PROMPT = """You are an expert Reinforcement Learning tutor helping students learn RL concepts interactively.
You explain concepts clearly with examples, use analogies (especially the dog-training and grid world examples from the lesson),
and are encouraging and patient. Keep responses concise (2-4 paragraphs) and educational.
Focus on the concepts covered: RL basics, the RL framework, policies, value functions, Bellman equation, and Q-learning.
When relevant, refer to the 5x5 Grid World example with Start at (0,0), Goal at (4,4), and traps at (0,3),(1,1),(2,2),(3,3)."""


def get_llm_response(provider: str, api_key: str, model: str,
                     user_message: str, context: str = "") -> str:
    """Get a response from the selected LLM provider."""
    full_message = f"Context: {context}\n\nStudent question: {user_message}" if context else user_message

    if provider == "Claude (Anthropic)":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": full_message}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error calling Claude: {str(e)}"

    elif provider == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_message}
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"

    elif provider == "Groq":
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_message}
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Groq: {str(e)}"

    elif provider == "Gemini (Google)":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
            response = model_obj.generate_content(full_message)
            return response.text
        except Exception as e:
            return f"Error calling Gemini: {str(e)}"

    return "Please configure an LLM provider in the sidebar to ask questions."
