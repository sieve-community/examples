import sieve
from pydantic import BaseModel

class LangIDResponse(BaseModel):
    language_code: str
    confidence: float

metadata = sieve.Metadata(
    description="Detect the language of a text with LangID, a lightweight language identification tool",
    code_url="https://github.com/sieve-community/examples/blob/main/language_classification",
    image=sieve.Image(
        url="https://miro.medium.com/v2/resize:fit:1400/1*hXcPKv1nTiKj7yzTdOlSoQ.gif"
    ),
    tags=["Text", "Language", "Translation"],
    readme=open("LANGID.README.md", "r").read(),
)

@sieve.Model(name="langid", python_packages=["langid"], metadata=metadata)
class Detector:
    def __setup__(self):
        from langid.langid import LanguageIdentifier, model
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    def __predict__(self, text: str) -> LangIDResponse:
        """
        :param text: text to detect language of
        :return: language code and confidence as a JSON object with keys `language_code` in ISO 639-1 format and `confidence` as a float
        """
        print("Classifying text:", text)
        lang, confidence = self.identifier.classify(text)
        print(f"Classified as {lang} with confidence {confidence}")
        return {
            "language_code": lang,
            "confidence": confidence
        }