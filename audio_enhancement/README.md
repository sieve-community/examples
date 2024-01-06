# Audio Enhancement

This function enhances audio by performing tasks such as upscaling and noise reduction. It accepts an audio file as input and returns an enhanced version of the same. The function provides three filter types to choose from:
* `upsample` - This filter upsamples the audio, making the speech sound clearer. It is useful when the audio is of low quality.
* `noise` - This filter dampens the background noise in the audio. It is useful when the audio has a lot of background noise that needs to be reduced.
* `all` - This filter performs both the tasks of upscaling and noise reduction. It first removes the background noise and then upsamples the audio.

## Usage

To use this function, you need to call the `enhance_audio` function with the following parameters:

* `audio`: An audio input (mp3 and wav supported)
* `filter_type`: Task to perform, one of ["upsample", "noise", "all"]
* `enhancement_steps`: Number of enhancement steps applied to the audio between 10 and 150. Higher values may improve quality but will take longer to process. Defaults to 50. Only applicable if `speed_boost` is False.
* `enhance_speed_boost`: If True, use a faster but more experimental model for audio enhancement. Depending on the type of audio files you're processing, it may work better at times. Defaults to False.
