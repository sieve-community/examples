# Transcript Analyzer 

## Overview
The Analyze Transcripts app offers a comprehensive solution for extracting valuable information from video or audio files. It automatically generates titles, chapters, summaries, and tags, enhancing the accessibility and discoverability of your media content. This tool is particularly useful for content creators, marketers, and educational purposes, providing insights and structured data that can be used to improve content engagement and understanding.

## Features
- **Title Generation**: Creates a concise and relevant title for your media file based on its content, with a maximum length of 10 words.
- **Chapter Generation**: Automatically divides the content into chapters, making it easier to navigate through different sections of the video or audio.
- **Summary Creation**: Produces a summary of the main points covered in the media file, with a customizable maximum sentence length.
- **Tagging**: Generates tags related to the content, aiding in searchability and categorization.
- **Audio Denoising**: Improves transcription accuracy by denoising audio tracks before processing.
- **Highlight Extraction**: Identifies and extracts highlights based on predefined search phrases, with adjustable maximum duration.

## Usage
### Input: Video or audio file.
### Parameters:
- `max_summary_length`: Maximum number of sentences in the summary. Default is 5.
- `max_title_length`: Maximum number of words in the title. Default is 10.
- `num_tags`: Number of tags to generate. Default is 5.
- `generate_chapters`: Option to generate chapters. Default is True.
- `denoise_audio`: Option to denoise audio for better transcription accuracy. Default is True.
- `generate_highlights`: Option to generate highlights based on specific phrases. Default is True.
- `highlight_search_phrases`: Topics or phrases for highlight extraction. Default is "Most likely to go viral, funniest".
- `max_highlight_duration`: Maximum duration of each highlight in seconds. Default is 30.
    - `relevance_score`: Measure of how relevant the highlight is with respect to the provided phrases. Computed by rating segments through GPT4.

## How It Works
Upon receiving a video or audio file, the app first converts the video to audio format (if necessary) and then runs a speech-to-text process to transcribe the content. It then processes the transcript to generate a title, chapters, a summary, and tags that accurately reflect the content with the help LLMs. Denoising can be performed before transcription to ensure higher accuracy. Additionally, the app can extract highlights based on specific topics or phrases, providing quick access to key moments.
