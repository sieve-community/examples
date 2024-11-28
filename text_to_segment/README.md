
# SAM2 text-to-segment
Simple demo app to enable text prompts for SAM2.

Have a look at these sieve functions to see how we use this building block!
- [focus effect](https://www.sievedata.com/functions/sieve-internal/sam2-focus)
- [callout effect](https://www.sievedata.com/functions/sieve-internal/sam2-callout)
- [color filter](https://www.sievedata.com/functions/sieve-internal/sam2-color-filter)
- [background blur](https://www.sievedata.com/functions/sieve-internal/sam2-blur)
- [selective color](https://www.sievedata.com/functions/sieve-internal/sam2-selective-color)
- [censorship](https://www.sievedata.com/functions/sieve-internal/sam2-pixelate)


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


