
# SAM2 text-to-segment
Simple demo app to enable text prompts for SAM2.

## Usage
Upload a video or a photo and name the object you want to track.


## Example
```python
sam = sieve.function.get("sieve/text-to-segment")
video_path = "duckling.mp4"
text_prompt = "duckling"

video = sieve.File(path=video_path)
sam_out = sam.run(video, text_prompt)
```


