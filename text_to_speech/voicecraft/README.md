# VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild
[![Paper](https://img.shields.io/badge/arXiv-2301.12503-brightgreen.svg?style=flat-square)](https://jasonppy.github.io/assets/pdfs/VoiceCraft.pdf)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://jasonppy.github.io/VoiceCraft_web/)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/pyp1/VoiceCraft_gradio)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IOjpglQyMTO2C3Y94LD9FY0Ocn-RJRg6?usp=sharing)

### TL;DR
VoiceCraft is a token infilling neural codec language model, that achieves state-of-the-art performance on both **speech editing** and **zero-shot text-to-speech (TTS)** on in-the-wild data including audiobooks, internet videos, and podcasts.

To clone or edit an unseen voice, VoiceCraft needs only a few seconds of reference.

## How to run inference

### TTS Mode
For simple inference using a reference voice use TTS Mode
1. pass a reference audio you want to imitate to reference_audio parameter and pass the text you want to synthesize to the input_text param
2. Generate similar sounding speech with your text.

```
audio = sieve.Audio(path="path/to/audio")
text = "text to synthesize "
voicecraft = sieve.function.get("ahanzala-bscs20seecs-seecs-edu-pk/voicecraft")
output = voicecraft.run(reference_audio = audio, input_text = text)
print(output.path)

```
### Edit Mode
For editing audios you can use this mode.
1. For edit mode change the mode = "Edit".
2. Editing mode allows for insertions, deletions and subsitutions.
3. For insertion pass the edit_start_time to the place where you want to insert words or sentence, edit_end_time should also be equal to edit_start_time in insertion.
4. For subsitution, pass the edit_start_time and edit_end_time of the word you want to replace, and pass the new word in the input_text
5. For deletions you may pass the starting and end time of the word to be deleted and pass ' ' in the reference string

Example:

```
audio = sieve.Audio(path="path/to/audio")
text = "text to subsitute "

voicecraft = sieve.function.get("ahanzala-bscs20seecs-seecs-edu-pk/voicecraft")
output = voicecraft.run(reference_audio = audio, input_text = text, mode= "Edit", edit_start_time = 1, edit_end_time = 2.4)
print(output.path)

```

### Long TTS
For many sentences or longer texts you can use this mode
1. Change mode to mode= "Long TTS"
2. You can specify how your sentences are seperated by a Newline or by Sentences using the split_text parameter
3. Generate

```
audio = sieve.Audio(path="path/to/audio/")
text = " this is sentence one \n this is sentence two \n this is sentence three "

voicecraft = sieve.function.get("ahanzala-bscs20seecs-seecs-edu-pk/voicecraft")
output = voicecraft.run(reference_audio = audio, input_text = text, mode= "Long TTS", split_text = "Newline")

```
