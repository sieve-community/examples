
def get_deepl_translation(text: str, source_language: str, target_language: str):
    import deepl
    import os
    deepl_api_key = os.getenv("DEEPL_API_KEY")
    if not deepl_api_key:
        raise ValueError("DEEPL_API_KEY environment variable not set")
    translator = deepl.Translator(deepl_api_key)
    if target_language == 'en':
        target_language = 'EN-US'
    if target_language == 'pt':
        target_language = 'PT-PT'
    result = translator.translate_text(text, source_lang=source_language, target_lang=target_language)
    return result.text

