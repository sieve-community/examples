import sieve

@sieve.function(
    name="video-captioner-combiner",
    gpu = False,
    python_packages=[
        "moviepy==1.0.3",
        "opencv-python==4.6.0.66",
        "uuid==1.30",
    ],
    python_version="3.8",
    iterator_input=True,
    persist_output=True
)
def caption_and_combine(videos, prompt_pairs) -> sieve.Video:
    from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
    import cv2
    import textwrap
    import uuid

    # Sort videos by global id
    videos = sorted(videos, key=lambda video: video.video_number)

    # Add captions
    images = []
    for v, prompt in zip(videos, prompt_pairs):
        # Add caption
        cap = cv2.VideoCapture(v.path)
        while cap.isOpened():
            # Capture frames in the video
            ret, frame = cap.read()
            if not ret:
                break

            # Add caption with textwrap
            font = cv2.FONT_HERSHEY_SIMPLEX
            wrapped_text = textwrap.wrap(prompt[0], width=30)
            x, y = 10, 40
            font_size = 1
            font_thickness = 2

            for i, line in enumerate(wrapped_text):
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

                gap = textsize[1] + 10

                y = int((frame.shape[0] + textsize[1]) / 2) + i * gap
                x = int((frame.shape[1] - textsize[0]) / 2)

                cv2.putText(frame, line, (x, y), font,
                            font_size, 
                            (255,255,0), 
                            font_thickness, 
                            lineType = cv2.LINE_AA)
                
            images.append(frame)
        
    clips = [ImageClip(m).set_duration(0.25) for m in images]
    video = concatenate_videoclips(clips)
    video_path = f"{uuid.uuid4()}.mp4"
    video.write_videofile(video_path, fps=30)
    return sieve.Video(path=video_path)
    