import sieve
import langcodes
from simple_tokenizers import *

metadata = sieve.Metadata(
    title="Text Translation",
    description="Reliable text translation through complex tokenization and sentence splitting.",
    code_url="https://github.com/sieve-community/examples/blob/main/translation/translate",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/translate-text-icon.png"
    ),
    tags=["Translation", "Text", "Showcase"],
    readme=open("README.md", "r").read(),
)

seamless = sieve.function.get("sieve/seamless_text2text")
langid = sieve.function.get("sieve/langid")

@sieve.function(
    name="translate",
    python_packages=["nltk", "langcodes[data]", "langdetect"],
    run_commands=[
        "python -m nltk.downloader punkt",
    ],
    metadata=metadata
)
def translate(
    text: str,
    source: str = 'auto',
    target: str = 'es'
) -> str:
    '''
    :param text: The text to translate
    :param source: The source language of the text in ISO 639-1 format. If 'auto', the language will be detected automatically.
    :param target: The target language of the text in ISO 639-1 format. Default is 'es' (Spanish).
    :return: The translated text
    '''
    from nltk.tokenize import sent_tokenize, word_tokenize
    from string import punctuation
    from langdetect import detect

    if source == 'auto':
        source = langid.run(text)["language_code"]
    
    source = source.lower()
    
    try:
        source_langcode = langcodes.standardize_tag(source)
        target_langcode = langcodes.standardize_tag(target)
        source_language = langcodes.Language.get(source_langcode)
        print("Identified source language as", source_language.display_name())
        target_language = langcodes.Language.get(target_langcode)
        print("Identified target language as", target_language.display_name())
    except langcodes.LanguageTagError as e:
        raise ValueError(f"Unsupported language code. {e}")

    try:
        sentences = sent_tokenize(text, language=source_language.display_name().lower())
    except:
        print("Failed to tokenize sentences using NLTK. Falling back to simple tokenizer.")
        sentences = universal_sentence_tokenizer(text)

    translations = []
    for sentence in sentences:
        # Remove punctuation and convert to lowercase
        try:
            words = word_tokenize(sentence)
        except:
            print("Failed to tokenize words using NLTK. Falling back to simple tokenizer.")
            words = universal_word_tokenizer(sentence)
        words_without_punctuation = [word.lower() for word in words if word not in punctuation]
        sentence_without_punctuation = " ".join(words_without_punctuation)

        # Sometimes the sentence is just punctuation
        if len(sentence_without_punctuation) <= 1:
            continue
        job = seamless.push(
            text=sentence_without_punctuation,
            source_language=source_language.to_alpha3(),
            target_language=target_language.to_alpha3()
        )
        translations.append(job)
    
    print(f"Translating {len(translations)} sentences...")
    translation_results = []
    for i, job in enumerate(translations, start=1):
        result = job.result()
        print(f"Translated sentence {i}/{len(translations)}")
        translation_results.append(result)
    
    print("Translation complete.")

    text = " ".join(translation_results)

    return {
        "translation": text,
        "sentences": [{"original": sentence, "translated": translation} for sentence, translation in zip(sentences, translation_results)]
    }
