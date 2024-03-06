import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole, GPTModelVersion
from pydantic import BaseModel, Field
import os

API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None or API_KEY == "":
    raise Exception("OPENAI_API_KEY environment variable not set")

def add_timecodes_to_chapters(chapters):
    chapters = [chapter.dict() for chapter in chapters]

    out_chapters = []
    for i, chapter in enumerate(chapters):
        out_chapters.append(
            {
                "title": chapter["title"],
                "timecode": f"{int(chapter['start_time'] // 3600):02d}:{int((chapter['start_time'] % 3600) // 60):02d}:{int(chapter['start_time'] % 60):02d}",
                "start_time": chapter["start_time"],
            }
        )
    return out_chapters
 
async def description_runner(
    transcript, max_num_sentences=5, max_num_words=10, num_tags=5
):
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
    - Ensure you reply with the right content, and not anything to do with the prompt.

    Respond with the following JSON schema:

    {json_schema}
    """

    gpt_json = GPTJSON[DescriptionSchema](api_key=API_KEY, model="gpt-3.5-turbo-16k")
    text = text = " ".join([segment["text"] for segment in transcript])

    max_num_tokens = 7000
    max_num_words = 3 * (max_num_tokens / 4)

    if text.count(" ") > max_num_words:
        print("Splitting transcript into multiple messages due to length for summary")
        messages = []
        current_message = ""
        for segment in transcript:
            if current_message.count(" ") > max_num_words:
                messages.append(current_message)
                current_message = ""
            current_message += f"{segment['text']} "
        messages.append(current_message)

        print(f"Split transcript into {len(messages)} messages")

        outputs = []
        for message in messages:
            payload = await gpt_json.run(
                messages=[
                    GPTMessage(
                        role=GPTMessageRole.SYSTEM,
                        content=SUMMARY_PROMPT,
                    ),
                    GPTMessage(role=GPTMessageRole.USER, content=f"Text: {message}"),
                ],
                format_variables={
                    "max_num_sentences": max_num_sentences,
                    "max_num_words": max_num_words,
                    "num_tags": num_tags,
                },
            )

            schema, transforms = payload

            outputs.append(schema)

        # now consolidate outputs in case there are duplicates or overlapping chapters due to splitting
        CONSOLIDATE_PROMPT = """
        You are a developer assistant where you only provide the code for a question. No explanation required. Write a simple json sample.

        Can you consolidate the summary, title, and tags from the previous messages into a single summary, title, and list of tags? Make sure to think step by step. First, think about the most important topics in the video and list them out in the order they appear. Then, think about how you would divide the video into chapters based on these topics. Finally, think about how you would title each chapter.

        The reason we are asking you to consolidate the summary, title, and tags is because the previous messages may have overlapping chapters due to splitting the transcript into multiple messages. We want to make sure that the summary, title, and tags are not overlapping.

        Please meet the following constraints:
        - The summary should cover all the key points and main ideas presented in the original text
        - The summary should be something that may follow the phrase "In this video..."
        - The summary should condense the information into a concise and easy-to-understand format
        - Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
        - The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.
        - Please limit your summary to {max_num_sentences} sentences.
        - Please limit your title to {max_num_words} words.
        - Please return {num_tags} tags that are most topical to the transcript.
        - Ensure you reply with the right content, and not anything to do with the prompt.

        Respond with the following JSON schema:

        {json_schema}
        """

        outputs = [output.dict() for output in outputs]

        outputs_str = "\n".join(
            [
                f"Title: {output['title']}\nSummary: {output['summary']}\nTags: {output['tags']}"
                for output in outputs
            ]
        )

        payload = await gpt_json.run(
            messages=[
                GPTMessage(
                    role=GPTMessageRole.SYSTEM,
                    content=CONSOLIDATE_PROMPT,
                ),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content=f"Text: {outputs_str}",
                ),
            ],
            format_variables={
                "max_num_sentences": max_num_sentences,
                "max_num_words": max_num_words,
                "num_tags": num_tags,
            },
        )

        schema, transforms = payload

        return schema.summary, schema.title, schema.tags
    else:
        payload = await gpt_json.run(
            messages=[
                GPTMessage(
                    role=GPTMessageRole.SYSTEM,
                    content=SUMMARY_PROMPT,
                ),
                GPTMessage(role=GPTMessageRole.USER, content=f"Text: {text}"),
            ],
            format_variables={
                "max_num_sentences": max_num_sentences,
                "max_num_words": max_num_words,
                "num_tags": num_tags,
            },
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
    - Please ensure the time of the chapter is marked as when the topic is first introduced based on the times you have been given as numbers
    - The chapters should include any adbreaks or other breaks in the video and name them accordingly "Ad Break"
    - The chapters should focus on macro-level topics, not micro-level details. This is very important. I cannot stress this enough.
    - The chapters should be evenly spaced throughout the video
    - Please be concise, and keep the chapter titles short and descriptive.

    Respond with the following JSON schema for chapters:

    {json_schema}

    Each chapter should have the following fields:
    - title: title of the chapter
    - start_time: start time of the chapter as a float in seconds
    """

    gpt_json = GPTJSON[ChaptersSchema](api_key=API_KEY, model="gpt-3.5-turbo-16k")
    text = " ".join([segment["text"] for segment in transcript])
    segment_info = [
        {
            "start_time": segment["start"],
            "end_time": segment["end"],
            "text": segment["text"],
        }
        for segment in transcript
    ]

    segments_str = "\n".join(
        [f"{segment['start_time']}: {segment['text']}" for segment in segment_info]
    )

    max_num_tokens = 7000
    max_num_words = 3 * (max_num_tokens / 4)
    if segments_str.count(" ") > max_num_words:
        print("splitting transcript into multiple messages due to length")
        # split into multiple messages with the same format
        messages = []
        current_message = ""
        for segment in segment_info:
            if current_message.count(" ") > max_num_words:
                messages.append(current_message)
                current_message = ""
            current_message += f"{segment['start_time']}: {segment['text']}\n"
        messages.append(current_message)

        print(f"split transcript into {len(messages)} messages")

        chapters = []
        for message in messages:
            payload = await gpt_json.run(
                messages=[
                    GPTMessage(
                        role=GPTMessageRole.SYSTEM,
                        content=PROMPT,
                    ),
                    GPTMessage(role=GPTMessageRole.USER, content=f"Transcript: {message}"),
                ]
            )

            schema, transforms = payload

            chapters.extend(schema.chapters)

        # now consolidate chapters in case there are duplicates or overlapping chapters due to splitting
        CONSOLIDATE_PROMPT = """
        You are a developer assistant where you only provide the code for a question. No explanation required. Write a simple json sample.

        Can you consolidate the chapters from the previous messages into a single list of chapters? Make sure to think step by step. First, think about the most important topics in the video and list them out in the order they appear. Then, think about how you would divide the video into chapters based on these topics. Finally, think about how you would title each chapter.

        The reason we are asking you to consolidate the chapters is because the previous messages may have overlapping chapters due to splitting the transcript into multiple messages. We want to make sure that the chapters are not overlapping.

        Please meet the following constraints:
        - The chapters should cover only the key points and main ideas presented in the original transcript
        - Please ensure the time of the chapter is marked as when the topic is first introduced based on the times you have been given as numbers
        - The chapters should include any adbreaks or other breaks in the video and name them accordingly "Ad Break"
        - The chapters should focus on macro-level topics, not micro-level details. This is very important. I cannot stress this enough.
        - The chapters should be evenly spaced throughout the video
        - Please be concise, and keep the chapter titles short and descriptive.

        Respond with the following JSON schema for chapters:

        {json_schema}

        Each chapter should have the following fields:
        - title: title of the chapter
        - start_time: start time of the chapter as a float in seconds
        """

        chapters_list_flattened = [
            {"title": chapter.title, "start_time": chapter.start_time}
            for chapter in chapters
        ]

        chapters_list_flattened = sorted(
            chapters_list_flattened, key=lambda x: x["start_time"]
        )

        chapters_str = "\n".join(
            [
                f"{chapter['start_time']}: {chapter['title']}"
                for chapter in chapters_list_flattened
            ]
        )

        payload = await gpt_json.run(
            messages=[
                GPTMessage(
                    role=GPTMessageRole.SYSTEM,
                    content=CONSOLIDATE_PROMPT,
                ),
                GPTMessage(
                    role=GPTMessageRole.USER,
                    content=f"Chapters: {chapters_str}",
                ),
            ]
        )

        schema, transforms = payload

        return add_timecodes_to_chapters(schema.chapters)
        
    else:
        payload = await gpt_json.run(
            messages=[
                GPTMessage(
                    role=GPTMessageRole.SYSTEM,
                    content=PROMPT,
                ),
                GPTMessage(role=GPTMessageRole.USER, content=f"Transcript: {segments_str}"),
            ]
        )

        schema, transforms = payload

        return add_timecodes_to_chapters(schema.chapters)
