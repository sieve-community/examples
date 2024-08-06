'''
Sieve workflow to generate a storybook video from a piece of writing.
'''

import sieve
from walker import StableDiffusionVideo
from caption_combine import caption_and_combine

# Creates a cleaned up list of sentences from a piece of writing
@sieve.function(name="prompt-to-script")
def prompt_to_script(prompt: str) -> list:
    script = prompt.split(".")
    script = [s.strip() for s in script if s.strip() != ""]
    script = [s + "." for s in script]
    return script

# Generates pairs of sentences from a list of sentences
@sieve.function(name="create-prompt-pairs")
def create_prompt_pairs(script: list) -> tuple:
    for i in range(len(script) - 1):
        yield (script[i], script[i + 1])

@sieve.workflow(name="storybook_generation")
def storybook_generation(prompt: str) -> sieve.Video:
    # Create a script (list of sentences) and pair them up
    print("Generating script and prompt pairs...")
    script = prompt_to_script(prompt)
    prompt_pairs = create_prompt_pairs(script)

    # Generate videos with StableDiffusionWalker
    print("Generating videos...")
    videos = StableDiffusionVideo()(prompt_pairs)

    # Return a captioned and concatenated video
    print("Generating storybook...")
    combined_video = caption_and_combine(videos, prompt_pairs)
    return combined_video
