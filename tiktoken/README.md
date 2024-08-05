# Tiktoken

`tiktoken` is a fast, open-source tokenizer developed by OpenAI. This app helps you count tokens in a text string, which is particularly useful for understanding the limits and costs associated with using OpenAI's GPT models.

## Key Features

- **Fast Tokenization**: Efficiently splits text into tokens, the units of text processed by GPT models.
- **Model Compatibility**: Supports multiple encoding schemes used by different OpenAI models.

## Supported Encodings

Different models use different encodings to convert text into tokens. `tiktoken` supports the following encodings:

| Encoding Name | OpenAI Models |
|---------------|---------------|
| cl100k_base   | gpt4o, gpt4o-mini, gpt-4, gpt-3.5-turbo, text-embedding-ada-002 |
| p50k_base     | Codex models, text-davinci-002, text-davinci-003 |
| r50k_base (gpt2) | GPT-3 models like davinci |
