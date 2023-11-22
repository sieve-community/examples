# Live HLS stream demo with VTT transcription

This demo shows the ABC News live stream with VTT transcription. The live transcription runs on Sieve and the VTT file is streaming to the video player.

Note: This is just to be meant as a demonstration. We currently save the VTT file locally and update the video player with the new VTT file every second. This is not a good practice for production.

## How to run
1. Sign up for a Sieve account and follow these [instructions](https://docs.sievedata.com/guide/examples/live-audio-transcription) to set up your Sieve project.
2. Clone this repo and run `python vtt_demo.py`
3. In a separate terminal, run `python -m http.server` to start a local server. Subtitles are not available if you open the HTML file directly.
4. Open [http://localhost:8000/vtt_demo.html](http://localhost:8000/vtt_demo.html) in your browser to see the live stream with VTT transcription. 

The subtitles may be out of sync with the video.  This has to do with the way the HTML video player renders HLS streams. In a permanent solution, the VTT file should be streamed to the video player directly.

## Demo video
Here's a demo of the system running with the browser + stream on the left and the Sieve function running in the top right.

https://github.com/sieve-community/examples/assets/6136843/5b691caa-f226-4d6e-8324-4d45d72ed1ab
