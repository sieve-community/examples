# Dubbing

Take a video or audio sample of a single person speaking and dub the audio in any language of your choosing. You can also lipsync the dub on the source material if it is a video.

**Note:** The processing time depends on length, and if a video on the resolution and length of the video but a general rule of thumb is that it takes 1 second per second of audio, and 12 seconds to generate a video with lipsyncing with default settings.

Some options to toggle include whether or not to refine the input and output audio, hyperparameters for speech generation, whether to downsample the video, and whether to lipsync the output.

For generating the text, we give the user an option to pick between the open source [xtts](https://www.sievedata.com/functions/sieve/xtts-v1), [ElevenLabs](https://www.sievedata.com/functions/sieve/elevenlabs_speech_synthesis), or [Play.ht](https://www.sievedata.com/functions/sieve/playht_speech_synthesis) Text-to-Speech models. The ElevenLabs and Play.ht models are recommended for better quality but requires an API key.

If you are using the ElevenLabs or Play.ht models, you can also choose to either clone the voice from the audio within the video itself or use a `voice_id` that you've already either created or is available in one of the platforms. If you choose to clone the voice, you can also choose to delete the voice after use.

You must enter the API keys and user IDs mentioned in the speech_synthesis functions under the ["Secrets" tab](https://www.sievedata.com/dashboard/settings/secrets) in your account settings if you want to use the API variants.

Tips for the input audio:
- Ensure there is only 1 speaker. If there is some noise it is ok, you can toggle the refinement options to denoise the audio if you would like.

## Options

- `source_file`: An audio or video input file to dub.
- `target_language`: The language to which the audio will be translated. Default is "spanish".
- `tts_model`: The Text-to-Speech model to use. Supported models are "xtts", "elevenlabs", and "playht". "elevenlabs" or "playht" are recommended for better quality but requires an ElevenLabs or PlayHT API key.
- `speech_stability`: A value between 0 and 1. Increasing this value can make the speech more expressive with output varying between re-generations. However, it can also lead to instabilities.
- `speech_similarity_boost`: A value between 0 and 1. Lower values are recommended if there are background artifacts present in the generated speech.
- `voice_id`: The ID of the voice to use. If none are set, the voice will be cloned from the source audio and used. This is only applicable if the `tts_model` is set to "elevenlabs" or "playht".
- `cleanup_voice_id`: Whether to delete the voice after use. This is only applicable if the `tts_model` is set to "elevenlabs" or "playht".
- `refine_source_audio`: Whether to refine the source audio using sieve/audio_enhancement.
- `low_resolution`: Whether to reduce the resolution of an input video to half of the original on each axis. Significantly speeds up inference. Defaults to False. Only applicable for video inputs.
- `low_fps`: Whether to reduce the fps of an input video to half of the original. Significantly speeds up inference. Defaults to False. Only applicable for video inputs.
- `enable_lipsyncing`: Whether to enable lip-syncing on the original video to the dubbed audio. Defaults to True. Only applicable for video inputs. Otherwise, audio is returned.


Currently, the languages supported end to end are: 
- English
- Spanish
- Chinese
- French
- Italian
- Portuguese
- Polish
- Turkish
- Russian
- Dutch
- Czech
- German
- Arabic

If using the Eleven Labs model for voice cloning, it additionally supports these languages:

- Korean
- Swedish
- Indonesian
- Vietnamese
- Filipino
- Ukrainian
- Greek
- Finnish
- Romanian
- Danish
- Bulgarian
- Malay
- Hungarian
- Norwegian
- Slovak
- Croatian
- Classic Arabic
- Tamil
- Hindi
