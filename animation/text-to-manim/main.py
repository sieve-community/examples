import sieve
from utils import *

metadata = sieve.Metadata(
    description="Generate educational and math animations from a text prompt using Manim.",
    code_url="https://github.com/sieve-community/examples/tree/main/animation/text-to-manim",
    image=sieve.Image(
        url="https://flyingframes.readthedocs.io/en/latest/_images/ch1_6_0.png"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name='text-to-manim',
    metadata=metadata,
    system_packages= [
        "ffmpeg",
        "sox",
        "libcairo2",
        "libcairo2-dev",
        "libgl1-mesa-dev",
        "texlive",
        "texlive-fonts-extra",
        "libpango1.0-dev"
    ],
    python_packages=[
        "manimpango",
        "manim",
        "anthropic",
    ],
    environment_variables=[
        sieve.Env(name="ANTHROPIC_API_KEY", description="Anthropic API Key")
    ],
)
def text_to_manim(prompt: str) -> str:
    """
    :param prompt: The prompt to generate an animation for
    :return: The generated animation
    """
    import anthropic
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise Exception("ANTHROPIC_API_KEY not found in environment variables. Please set ANTHROPIC_API_KEY to your Anthropic API key.")
    
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    print(f"Generating manim code for prompt: {prompt}")

    response = client.messages.create(
        model="claude-3-opus-20240229",
        system=GPT_SYSTEM_INSTRUCTIONS,
        messages=[
            {"role": "user", "content": wrap_prompt(prompt)}
        ],
        max_tokens=1200,
        temperature=0,
    )
    print(response.content[0].text)
    code_response = extract_construct_code(
      extract_code(response.content[0].text)
    )
    
    print("Writing code to file...")
    try:
        with open("GenScene.py", "w") as f:
            fcontents = create_file_content(code_response)
            f.write(fcontents)
    except:
        raise Exception("Error writing file")
    
    try:
        print("Rendering scene with manim...")
        import GenScene
        import importlib
        importlib.reload(GenScene)
        scene = GenScene.GenScene()
        scene.render()
    except:
        raise Exception("Error rendering scene")
    
    return sieve.File(path=scene.renderer.file_writer.movie_file_path)

if __name__ == '__main__':
    text_to_manim('Draw a circle with radius 2 and color red.')