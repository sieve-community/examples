import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole, GPTModelVersion
from pydantic import BaseModel, Field
import os

def get_api_key():
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY is None or API_KEY == "":
        raise Exception("OPENAI_API_KEY environment variable not set")
    return API_KEY

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

    gpt_json = GPTJSON[DescriptionSchema](api_key=get_api_key(), model="gpt-3.5-turbo-16k")
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

    gpt_json = GPTJSON[ChaptersSchema](api_key=get_api_key(), model="gpt-3.5-turbo-16k")
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

async def highlight_prompt_handler(highlights):
    class HighlightPrompt(BaseModel):
        system_prompt: str = Field(description="A prompt for creating video highlights")

    HIGHLIGHTS_SYSTEM_PROMPT = """

    Write a system prompt that will be used to generate highlights for a video. 

    - The prompt should be designed to generate highlights for a video by scoring segments of the video transcript out of 100.
    - The input to the system prompt will be a list of segments from the video, where each segment is a short piece of text from the transcript.
    - The prompt must be concise and must mention the need for scoring each segment.

    Respond with the following JSON schema:

    {json_schema}
    """
    gpt_json = GPTJSON[HighlightPrompt](get_api_key(), model="gpt-4-turbo-preview")
    payload = await gpt_json.run(
        messages=[
            GPTMessage(role=GPTMessageRole.SYSTEM, content=HIGHLIGHTS_SYSTEM_PROMPT),
            GPTMessage(role=GPTMessageRole.USER, content=f"generate prompt based on: {highlights}"),
        ]
    )
    return payload.response.system_prompt

async def highlight_titles_handler(highlights, summary):
    class HighlightTitles(BaseModel):
        titles: list[str] = Field(description="List of titles for the video highlights")

    HIGHLIGHTS_TITLES_PROMPT = """

    Write a list of titles for the video highlights based on the segments of the video transcript given its summary. 

    - The titles should be concise and descriptive of the content of the segment.
    - Each title should be a short phrase or sentence that captures the essence of the segment.
    - The titles should be engaging and informative, providing a clear idea of what the segment is about.
    - Please ensure that the titles are relevant to the content of the segment and accurately represent the information presented.
    - Respond with the following JSON schema:

    {json_schema}
    """
    gpt_json = GPTJSON[HighlightTitles](get_api_key(), model="gpt-4-turbo-preview")
    payload = await gpt_json.run(
        messages=[
            GPTMessage(role=GPTMessageRole.SYSTEM, content=HIGHLIGHTS_TITLES_PROMPT),
            GPTMessage(role=GPTMessageRole.USER, content=f"""
                        video's summary: {summary}               
                        generate titles based on: {highlights}
                        """),
        ]
    )
    return payload.response.titles

async def process_batch(batch, system_prompt):
    class HighlightSchema(BaseModel):
        highlights_scores: list[int]

    system_prompt = system_prompt + " Respond with the following JSON schema: {json_schema}"

    gpt_json = GPTJSON[HighlightSchema](get_api_key(), model="gpt-4-turbo-preview")
    payload = await gpt_json.run(
        messages=[
            GPTMessage(role=GPTMessageRole.SYSTEM, content=system_prompt),
            GPTMessage(role=GPTMessageRole.USER, content=str(batch)),
        ]
    )
    return payload.response.highlights_scores

async def highlight_runner(gpt_input, highlights):
    class HighlightPrompt(BaseModel):
        system_prompt: str = Field(description="A prompt for creating video highlights")

    HIGHLIGHTS_SYSTEM_PROMPT = """

    Write a system prompt that will be used to generate highlights for a video. 

    - The prompt should be designed to generate highlights for a video by scoring segments of the video transcript out of 100.
    - The input to the system prompt will be a list of segments from the video, where each segment is a short piece of text from the transcript.
    - The prompt must be concise and must mention the need for scoring each segment.
    - The prompt should encourage using other segments to score the current segment.

    Respond with the following JSON schema:

    {json_schema}
    """
    gpt_json = GPTJSON[HighlightPrompt](get_api_key(), model="gpt-4-turbo-preview")
    payload = await gpt_json.run(
        messages=[
            GPTMessage(role=GPTMessageRole.SYSTEM, content=HIGHLIGHTS_SYSTEM_PROMPT),
            GPTMessage(role=GPTMessageRole.USER, content=f"generate prompt based on: {highlights}"),
        ]
    )
    system_prompt = payload.response.system_prompt

    batch_size = 20
    scores = []
    tasks = []

    for i in range(0, len(gpt_input), batch_size):
        batch = gpt_input[i:i+batch_size]
        tasks.append(process_batch(batch, system_prompt))

    results = await asyncio.gather(*tasks)
    for result in results:
        scores.extend(result)

    return scores

def create_detailed_highlights(segments, max_duration):
    def generate_sequences(segments):
        n = len(segments)
        for start_idx in range(n):
            total_duration = 0
            sequence = []
            for end_idx in range(start_idx, n):
                if total_duration + segments[end_idx]["duration"] <= max_duration:
                    sequence.append(segments[end_idx])
                    total_duration += segments[end_idx]["duration"]
                else:
                    break
                yield sequence.copy()
    
    sequences = list(generate_sequences(segments))
    sequences_with_scores = [(seq, sum(item["score"] for item in seq)) for seq in sequences]
    sequences_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    used_segments = set()
    detailed_highlights = []
    
    for sequence, score in sequences_with_scores:
        if not any(segment["text"] in used_segments for segment in sequence):
            highlight = {
                "start_time": sequence[0]["start_time"],
                "end_time": sequence[-1]["end_time"],
                "duration": sequence[-1]["end_time"] - sequence[0]["start_time"],
                "transcript": " ".join(segment["text"] for segment in sequence),
                "relevance_score": score,
            }
            detailed_highlights.append(highlight)
            used_segments.update(segment["text"] for segment in sequence)
    
    # Sort the highlights based on their cumulative score in descending order making sure they are at least half the max duration
    detailed_highlights = [highlight for highlight in detailed_highlights if highlight["duration"] >= seconds_to_timestamp(max_duration) / 2]

    print("det_highlights", detailed_highlights)

    # remove duration from highlights
    for highlight in detailed_highlights:
        del highlight["duration"]

    detailed_highlights.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return detailed_highlights

def compute_scores(extended_dict, scores, max_duration, summary):
    # Update score in extended_dict
    for index, score in enumerate(scores):
        extended_dict[index]['score'] = score

    # Sorting window_data by start_time
    window_data = sorted(extended_dict.values(), key=lambda x: x['start_time'])
    optimal_windows = create_detailed_highlights(window_data, max_duration)
    
    # Sort and filter optimal_windows
    optimal_windows.sort(key=lambda x: x['relevance_score'], reverse=True)
    slice_size = len(optimal_windows) // 4 if len(optimal_windows) > 45 else len(optimal_windows) // 3
    optimal_windows = optimal_windows[:slice_size]

    # Adjust scores to be relative to the highest score
    max_score = max(window['relevance_score'] for window in optimal_windows)
    for window in optimal_windows:
        window['relevance_score'] = round(window['relevance_score'] / max_score * 100, 2)

    # add titles and timestamps to optimal_windows and remove transcript
    for i, window in enumerate(optimal_windows):
        title = asyncio.run(highlight_titles_handler([window['transcript']], summary))[0]
        del window['transcript']
        window['title'] = title
        window['start_time'] = window['start_time']
        window['end_time'] = window['end_time']

    return optimal_windows

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
