import sieve
from typing import Literal

metadata = sieve.Metadata(
    description="Count OpenAI tokens with tiktoken, manage limits and costs!",
    code_url="https://github.com/sieve-community/examples/blob/main/tiktoken",
    image=sieve.Image(
        url="https://avatars.githubusercontent.com/u/14957082?v=4&s=160"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="tiktoken",
    python_packages=["tiktoken"],
    metadata=metadata
)
def calculate_tokens(
    text: str, 
    encoding: Literal["cl100k_base","p50k_edit","p50k_base", "r50k_base","gpt2"] = "cl100k_base"
    ):
    '''
    :param text: The text to calculate tokens for
    :param encoding: The encoding to use for the tokens. For "gpt4" , "gpt-3.5-turbo","gpt-4o" ,"gpt-4o-mini" use cl100k_base
    :return: The number of tokens in the text
    '''
    import tiktoken
    encoding = tiktoken.get_encoding(encoding)
    num_tokens = len(encoding.encode(text))
    return num_tokens


