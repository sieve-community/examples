import sieve

metadata = sieve.Metadata(
    title="Generate Content Highlights",
    description="Generate and render video or audio highlights for long-form content based on search phrases.",
    code_url="https://github.com/sieve-community/examples/blob/main/video_editing/highlights",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/highlights-icon.webp"
    ),
    tags=["Video", "Highlights", "Clips", "Showcase"],
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="highlights",
    system_packages=["ffmpeg"],
    python_packages=["moviepy"],
    metadata=metadata,
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key")
    ],
)
def highlights(
    file: sieve.File,
    render_clips: bool = True,
    highlight_search_phrases: str = "most viral moments",
):
    """
    :param file: The video or audio file to process
    :param render_clips: If True, the function will render the clips and return the clips as files. If False, the function will just return the metadata of the highlights.
    :param highlight_search_phrases: The search phrases to use to generate highlights.
    """
    import os
    import shutil
    temp_dir = "temp"
    # Ensure temp directory is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    video_transcript_analyzer = sieve.function.get("sieve/video_transcript_analyzer")
    print("Processing file...")
    output = video_transcript_analyzer.run(
        file,
        generate_highlights=True,
        generate_chapters=False,
        denoise_audio=False,
        highlight_search_phrases=highlight_search_phrases,
    )

    try:
        # last output is the highlights
        highlights = list(output)[-1]["highlights"]
    except KeyError:
        raise ValueError("No highlights found in the output. Please ensure your file contains audio and is long enough to generate highlights.")

    from moviepy.editor import VideoFileClip

    if len(highlights) == 0:
        print("No highlights found.")

    if not render_clips:
        for highlight in highlights:
            yield highlight
    else:
        print("Rendering clips...")
        is_audio = file.path.endswith(('.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'))
        if is_audio:
            from moviepy.editor import AudioFileClip
            media_clip = AudioFileClip(file.path)
        else:
            from moviepy.editor import VideoFileClip
            media_clip = VideoFileClip(file.path)
        
        count = 0
        for highlight in highlights:
            print(f"Rendering clip {count}...")
            start_time = highlight["start_time"]
            end_time = highlight["end_time"]
            clip = media_clip.subclip(start_time, end_time)
            clip_path = os.path.join(temp_dir, f"highlight_{count}.mp4" if not is_audio else f"highlight_{count}.mp3")
            if is_audio:
                clip.write_audiofile(clip_path)
            else:
                clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
            yield sieve.File(path=clip_path), highlight
            count += 1

    return highlights

if __name__ == "__main__":
    file = sieve.File(url="https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/c32439cc-81d6-4023-b2df-2003f41abe3f/1fd41772-673f-4883-999d-19c5fa9372b9-input-file.mp4")
    for out in highlights(file):
        print(out)