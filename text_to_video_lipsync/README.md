# Text to Video Lipsyncing

This app takes in a video file and a piece of text and outputs a video that makes it look like the person in the video is saying the text.

For generating the text, we give the user an option to pick between the open source [xtts](https://www.sievedata.com/functions/sieve/xtts-v1), [ElevenLabs](https://www.sievedata.com/functions/sieve/elevenlabs_speech_synthesis), or [Play.ht](https://www.sievedata.com/functions/sieve/playht_speech_synthesis) Text-to-Speech models. The ElevenLabs and Play.ht models are recommended for better quality but requires an API key.

If you are using the ElevenLabs or Play.ht models, you can also choose to either clone the voice from the audio within the video itself or use a `voice_id` that you've already either created or is available in one of the platforms. If you choose to clone the voice, you can also choose to delete the voice after use.

You must enter the API keys and user IDs mentioned in the speech_synthesis functions under the ["Secrets" tab](https://www.sievedata.com/dashboard/settings/secrets) in your account settings if you want to use the API variants.

## Options

- `source_video`: The video file to lip-sync.
- `text`: The text that the person in the video will appear to speak.
- `tts_model`: The Text-to-Speech model to use. Supported models are "xtts", "elevenlabs", and "playht". "elevenlabs" or "playht" are recommended for better quality but requires an ElevenLabs API key.
- `speech_stability`: A value between 0 and 1. Increasing this value can make the speech more expressive with output varying between re-generations. However, it can also lead to instabilities.
- `speech_similarity_boost`: A value between 0 and 1. Lower values are recommended if there are background artifacts present in the generated speech.
- `voice_id`: The ID of the voice to use. If none are set, the voice will be cloned from the source audio and used. This is only applicable if the `tts_model` is set to "elevenlabs" or "playht".
- `cleanup_voice_id`: Whether to delete the voice after use. This is only applicable if the `tts_model` is set to "elevenlabs" or "playht".
- `refine_source_audio`: Whether to refine the source audio using sieve/audio_enhancement.
- `refine_target_audio`: Whether to refine the generated target audio using sieve/audio_enhancement.
- `low_resolution`: Whether to reduce the resolution of the output video to half of the original on each axis; significantly speeds up inference.
- `low_fps`: Whether to reduce the fps of the output video to half of the original; significantly speeds up inference.