from openai import OpenAI
from pydantic import BaseModel, Field
import os
import instructor

class TranslationOutput(BaseModel):
    translated_text: str = Field(description="The translated text")
    source_language: str = Field(description="The source language of the text in ISO 639-1 format")
    target_language: str = Field(description="The target language of the text in ISO 639-1 format")

def get_translation(text: str, source_language: str, target_language: str, llm_backend: str = "openai") -> TranslationOutput:
    if llm_backend == "mixtral":
        API_KEY = os.getenv("TOGETHERAI_API_KEY")
    else:
        API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY or API_KEY == "":
        raise Exception("OPENAI_API_KEY or TOGETHERAI_API_KEY environment variable not set")

    if llm_backend == "mixtral":
        client = OpenAI(api_key=API_KEY, base_url="https://api.together.xyz/v1")
    else:
        client = OpenAI(api_key=API_KEY)

    client = instructor.patch(client)

    model = "gpt-4-turbo-preview" if llm_backend == "openai" else "mistralai/Mixtral-8x7B-Instruct-v0.1"

    return client.chat.completions.create(
        model=model,
        response_model=TranslationOutput,
        messages=[
            {"role": "system", "content": f"Translate the following text from {source_language} to {target_language}. Ensure to stay true to the original meaning and keep the translation natural."},
            {"role": "user", "content": text}
        ],
        max_retries=3
    )

if __name__ == "__main__":
    print(get_translation("Hello, world!", "en", "fr"))
