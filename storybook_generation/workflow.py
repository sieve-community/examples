'''
Storybook generation workflow
'''

import sieve
from walker import StableDiffusionVideo
from combiner import combiner

# TODO: make this use an LLM
@sieve.function(name="prompt-to-script")
def prompt_to_script(prompt: str) -> list:
    script = prompt.split(".")
    script = [s.strip() for s in script if s.strip() != ""]
    script = [s + "." for s in script]
    return script


@sieve.function(name="create-prompt-pairs")
def create_prompt_pairs(script: list) -> tuple:
    for i in range(len(script) - 1):
        yield (script[i], script[i + 1])


@sieve.workflow(name="storybook_generation")
def storybook_generation(prompt: str) -> sieve.Video:
    script = prompt_to_script(prompt)
    prompt_pair = create_prompt_pairs(script)
    videos = StableDiffusionVideo()(prompt_pair)
    combined_video = videos #combiner(videos)
    return combined_video
