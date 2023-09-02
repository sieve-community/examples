import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole, GPTModelVersion
from pydantic import BaseModel, Field
import os

API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    raise Exception("OPENAI_API_KEY environment variable not set")

async def description_runner(transcript, max_num_sentences=5, max_num_words=10, num_tags=5):
    class DescriptionSchema(BaseModel):
        title: str = Field(description="Title of the video")
        summary: str = Field(description="Summary of the video")
        tags: list[str] = Field(description="Tags of the video")

    SUMMARY_PROMPT = """
    You are a developer assistant where you only provide the code for a question. No explanation required. Write a simple json sample.

    Can you provide a comprehensive title, summary, and tags of the given transcript? Please meet the following constraints:
    - The summary should cover all the key points and main ideas presented in the original text
    - The summary should be something that may follow the phrase "In this video..."
    - The summary should condense the information into a concise and easy-to-understand format
    - Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
    - The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.
    - Please limit your summary to {max_num_sentences} sentences.
    - Please limit your title to {max_num_words} words.
    - Please return {num_tags} tags that are most topical to the transcript.

    Respond with the following JSON schema:

    {json_schema}
    """

    gpt_json = GPTJSON[DescriptionSchema](api_key = API_KEY, model = "gpt-3.5-turbo-16k")
    text = text = " ".join([segment["text"] for segment in transcript])
    payload = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SUMMARY_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=f"Text: {text}"
            )
        ],
        format_variables={"max_num_sentences": max_num_sentences, "max_num_words": max_num_words, "num_tags": num_tags}
    )

    schema, transforms = payload

    return schema.summary, schema.title, schema.tags

async def chapter_runner(transcript):
    class Chapter(BaseModel):
        title: str = Field(description="Title of the video chapter")
        start_time: float = Field(description="Start time of the video chapter")

    class ChaptersSchema(BaseModel):
        chapters: list[Chapter] = Field(description="List of chapters")

    PROMPT = """
    You are a developer assistant where you only provide the code for a question. No explanation required. Write a simple json sample.

    Can you provide a list of chapter titles with start and end times for the given transcript? Make sure to think step by step. First, think about the most important topics in the video and list them out in the order they appear. Then, think about how you would divide the video into chapters based on these topics. Finally, think about how you would title each chapter.

    Please meet the following constraints:
    - The chapters should cover only the key points and main ideas presented in the original transcript
    - The chapters should focus on macro-level topics, not micro-level details. This is very important. I cannot stress this enough.
    - The chapter `start_time` values should NEVER EVER be within 15 seconds of each other.
    - The chapters should be evenly spaced throughout the video
    - The chapters should condense the information into concise, important topical divides and avoiding any unnecessary information or repetition
    - The number of chapters should be appropriate for the length and complexity of the original transcript, providing a clear and accurate overview without omitting any important information.
    - Please be concise, and keep the chapter titles short and descriptive.

    Respond with the following JSON schema for chapters:

    {json_schema}

    Each chapter should have the following fields:
    - title: title of the chapter
    - start_time: start time of the chapter as a float in seconds
    """

    gpt_json = GPTJSON[ChaptersSchema](api_key = API_KEY, model = "gpt-3.5-turbo-16k")
    text = " ".join([segment["text"] for segment in transcript])
    segment_info = [{"start_time": segment["start"], "end_time": segment["end"], "text": segment["text"]} for segment in transcript]

    segments_str = "\n".join([f"{segment['start_time']}: {segment['text']}" for segment in segment_info])
    payload = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=f"Transcript: {segments_str}"
            )
        ]
    )

    schema, transforms = payload

    return schema.chapters
