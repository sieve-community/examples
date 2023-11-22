import sieve

live_audio_transcriber = sieve.function.get("sieve/live_speech_transcriber")
stream = "https://content.uplynk.com/channel/3324f2467c414329b3b0cc5cd987b6be.m3u8"
output_file_path = "out/output.vtt"


def sec_to_timecode(sec):
    hours, remainder = sec // 3600, sec % 3600
    minutes, seconds = remainder // 60, remainder % 60
    timecode = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
    return timecode


with open(output_file_path, "w") as vtt_file:
    vtt_file.write("WEBVTT\n\n")

    for transcript in live_audio_transcriber.run(stream):
        start, end = sec_to_timecode(transcript["start"]), sec_to_timecode(
            transcript["end"]
        )
        vtt_file.write(f"{start} --> {end}\n")
        vtt_file.write(f"{transcript['text']}\n\n")
        vtt_file.flush()

        print(f"{start} --> {end}")
        print(f"{transcript['text']}\n\n")
