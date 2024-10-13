

Recently, Google released [Illuminate](https://illuminate.google.com/home), an experimental tool for generating engaging audio content (podcast-esque) from research papers and books.

From the web page:
> Illuminate is an experimental technology that uses AI to adapt content to your learning preferences. Illuminate generates audio with two AI-generated voices in conversation, discussing the key points of select papers. Illuminate is currently optimized for published computer science academic papers.
> As an experimental product, the generated audio with two AI-generated voices in conversation may not always perfectly capture the nuances of the original research papers. Please be aware that there may be occasional errors or inconsistencies and that we are continually iterating to improve the user experience.

While they don't go into detail on its inner workings, it's reasonable to believe that it's a pipeline or some sort with two parts:
- A text-based language model pipeline to extract information and write a transcript
- An audio pipeline to read the transcript and make a dialogue

The text-based pipeline would be responsible for taking in a PDF or ebook, and extracting key information using a technique like RAG, then generating talking points and dialogue from that information.

The audio examples on the Illuminate web page tend to follow an interview-style format along the lines of 
- introduce the content
- question
- in depth response
- reaction or comment, followup question

The examples are pleasant, factual, and use analogies to explain the content in a pleasing way. Designing a language model program to turn raw information into a human-like dialogue is a tricky task, and outside of the scope of this post!

Here we're going to attack the second part of the pipeline: generating dialogue from the conversation. At the end of the day, you'll probably end up with an openai chat schema with a conversation in it. Something that looks like this:
```python
    messages = [
        {"role": "user", "message": "Hello, how are you?"},
        {"role": "assistant", "message": "I'm doing well, thank you for asking."},
        {"role": "user", "message": "What is the capital of France?"},
        {"role": "assistant", "message": "The capital of France is Paris."},
    ]
```

The task is to generate a dialogue with two voices from the dialogue. The easy part of the task is to
- clone two voices
- read "user" messages in one voice, read "assistant" messages in the other
- add pauses in between dictations

The harder part of the task is to _contextually_ adjust
- intonation
- speed
- pauses

We'll accomplish this super easily with `sieve/tts`, which narrates text given a reference voice, and gives you control over emotion, pacing, and other granular parameters.

For each message, we'll
- predict a reasonable emotion and pace with an llm 






