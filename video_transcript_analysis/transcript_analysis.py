import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole, GPTModelVersion
from pydantic import BaseModel, Field
import os

def get_api_key():
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY is None or API_KEY == "":
        raise Exception("OPENAI_API_KEY environment variable not set")
    return API_KEY

def add_timecodes_to_highlights(highlights):
    highlights = [highlight.dict() for highlight in highlights]

    out_highlights = []

    for highlight in highlights:
        duration = highlight["end_time"] - highlight["start_time"] + 1
        highlight_dict = {
            "title": highlight["title"],
            "score": highlight["score"],
            "start_time": highlight["start_time"],
            "end_time": highlight["end_time"] + 1,
            "start_timecode": seconds_to_timestamp(highlight["start_time"]),
            "end_timecode": seconds_to_timestamp(highlight["end_time"] + 1),
            "duration": duration 
        }
        out_highlights.append(highlight_dict)
    
    return out_highlights

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

    gpt_json = GPTJSON[DescriptionSchema](api_key=get_api_key(), model="gpt-4o-2024-05-13", model_max_tokens=4095)
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

            outputs.append(payload.response)

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

        return payload.response.summary, payload.response.title, payload.response.tags
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

        return payload.response.summary, payload.response.title, payload.response.tags


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

    gpt_json = GPTJSON[ChaptersSchema](api_key=get_api_key(), model="gpt-4o-2024-05-13", model_max_tokens=4095)
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

            chapters.extend(payload.response.chapters)

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

        return add_timecodes_to_chapters(payload.response.chapters)
        
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

        return add_timecodes_to_chapters(payload.response.chapters)  
async def process_segments_in_batches(transcript_segments, highlight_phrases, media_length):
    BATCH_SIZE = 500
    async def process_batch(batch):
        return await highlight_generator_gpt4(batch, highlight_phrases)

    # Split the input list into batches
    batches = [transcript_segments[i:i + BATCH_SIZE] for i in range(0, len(transcript_segments), BATCH_SIZE)]
    
    # Use asyncio.gather to call the function concurrently for all batches
    batch_results = await asyncio.gather(*(process_batch(batch) for batch in batches))

    # Merge results from all batches
    merged_highlights = []
    for batch_result in batch_results:
        merged_highlights.extend(batch_result)

    # Separate highlights into two groups based on their duration
    short_highlights = [highlight for highlight in merged_highlights if highlight['duration'] <= 180]
    long_highlights = [highlight for highlight in merged_highlights if highlight['duration'] > 180]

    # Sort the short highlights by score in descending order
    short_highlights_sorted = sorted(short_highlights, key=lambda x: x['score'], reverse=True)

    # Sort the long highlights by duration in ascending order
    long_highlights_sorted = sorted(long_highlights, key=lambda x: x['duration'])

    # Concatenate the sorted short and long highlights
    final_sorted_highlights = short_highlights_sorted + long_highlights_sorted

    for i in range(len(final_sorted_highlights)):
        if final_sorted_highlights[i]['start_time'] < 0:
            final_sorted_highlights[i]['start_time'] = 0
        if final_sorted_highlights[i]['end_time'] > media_length:
            final_sorted_highlights[i]['end_time'] = media_length - 0.01
        
        final_sorted_highlights[i]['start_timecode'] = seconds_to_timestamp(final_sorted_highlights[i]['start_time'])
        final_sorted_highlights[i]['end_timecode'] = seconds_to_timestamp(final_sorted_highlights[i]['end_time'])
        final_sorted_highlights[i]['duration'] = final_sorted_highlights[i]['end_time'] - final_sorted_highlights[i]['start_time']

    return final_sorted_highlights

async def highlight_generator_gpt4(transcript_segments, highlight_phrases):
    class Highlight(BaseModel):
        title: str = Field(description="Title of the Video highlight")
        start_time: float = Field(description="Start time of the video highlight")
        end_time: float = Field(description="End time of the video highlight")
        score: float = Field(description="Score of the video highlight")

    class HighlightSchema(BaseModel):
        chapters: list[Highlight] = Field(description="List of Highlights")

    PROMPT = """
    You are a developer assistant where you only provide the code for a question. No explanation required. Write a simple json sample.
    Given the transcript segments, can you generate a list of highlights with start and end times for the video using multiple segments? Please meet the following constraints:

    Please meet the following constraints:
    - The highlights should be a direct part of the video and should not be out of context
    - The highlights should be interesting and clippable, providing value to the viewer
    - The highlights should not be too short or too long, but should be just the right length to convey the information
    - The highlights should include more than one segment to provide context and continuity
    - The highlights should not cut off in the middle of a sentence or idea
    - The user provided highlight phrases should be used to generate the highlights
    - The highlights should be based on the relevance of the segments to the highlight phrases
    - The highlights should be scored out of 100 based on the relevance of the segments to the highlight phrases

    Respond with the following JSON schema for highlights:

    {json_schema}

    Each highlight should have the following fields:
    - title: title of the highlight
    - start_time: start time of the highlight as a float in seconds
    - end_time: end time of the highlight as a float in seconds
    - score: score of the highlight as a float out of 100
    """

    input_segments = ""

    for segment in transcript_segments:
        input_segments += f"segment start time: {segment['start']}, segment end time: {segment['end']}, segment text: {segment['text']}\n"

    gpt_json = GPTJSON[HighlightSchema](api_key=get_api_key(), model="gpt-4o-2024-05-13", model_max_tokens=4095)
    payload = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=f"""
                highlight phrases: {highlight_phrases},
                
                Transcript: {input_segments}""",
            ),
        ]
    )

    return add_timecodes_to_highlights(payload.response.chapters)

## Utils
from datetime import timedelta

def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    timestr = f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}.{milliseconds:03d}"
    hours, minutes, seconds = map(float, timestr.split(':'))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)
