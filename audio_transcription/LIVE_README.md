# Live Speech Transcription + Translation

This app takes in a URL to a live stream of audio and transcribes + translates it. The URL just needs to point to a stream that is supported by FFMPEG. This includes RTMP streams, HLS streams, and [other formats](https://ffmpeg.org/ffmpeg-formats.html). Latency is anywhere between 600ms to 3s depending on the stream with translation adding an additional 1s to 2s.

The public implementation of this app limits you to a minute of streamed transcripts. If you want to run your stream indefinitely, please follow the instructions in our docs [here](https://docs.sievedata.com/guide/examples/live-audio-transcription).

You can find a list of supported language codes here:
* `en` - English
* `zh` - Chinese
* `de` - German
* `es` - Spanish
* `ru` - Russian
* `ko` - Korean
* `fr` - French
* `ja` - Japanese
* `pt` - Portuguese
* `tr` - Turkish
* `pl` - Polish
* `ca` - Catalan
* `nl` - Dutch
* `ar` - Arabic
* `sv` - Swedish
* `it` - Italian
* `id` - Indonesian
* `hi` - Hindi
* `fi` - Finnish
* `vi` - Vietnamese
* `he` - Hebrew
* `uk` - Ukrainian
* `el` - Greek
* `ms` - Malay
* `cs` - Czech
* `ro` - Romanian
* `da` - Danish
* `hu` - Hungarian
* `ta` - Tamil
* `no` - Norwegian
* `th` - Thai
* `ur` - Urdu
* `hr` - Croatian
* `bg` - Bulgarian
* `lt` - Lithuanian
* `cy` - Welsh
* `sk` - Slovak
* `te` - Telugu
* `bn` - Bengali
* `sr` - Serbian
* `sl` - Slovenian
* `kn` - Kannada
* `et` - Estonian
* `mk` - Macedonian
* `eu` - Basque
* `is` - Icelandic
* `hy` - Armenian
* `bs` - Bosnian
* `kk` - Kazakh
* `gl` - Galician
* `mr` - Marathi
* `pa` - Punjabi
* `km` - Khmer
* `sn` - Shona
* `yo` - Yoruba
* `so` - Somali
* `af` - Afrikaans
* `ka` - Georgian
* `be` - Belarusian
* `tg` - Tajik
* `sd` - Sindhi
* `gu` - Gujarati
* `am` - Amharic
* `lo` - Lao
* `nn` - Nynorsk
* `mt` - Maltese
* `my` - Myanmar
* `tl` - Tagalog
* `as` - Assamese
* `jw` - Javanese
* `yue` - Cantonese
