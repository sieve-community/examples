import sieve
from typing import Literal

metadata = sieve.Metadata(
    title="Florence-2",
    description="A visual language foundation model that can perform a variety of image and video question-answer tasks, such as object detection, image captioning, segmentation, and OCR.",
    code_url="https://github.com/sieve-community/examples/blob/main/visual_qa/florence-2",
    image=sieve.Image(
        url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    ),
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name='florence-2',
    python_version='3.10',
    python_packages=[
        'torch==2.3.1',
        'transformers',
        'pillow',
        'einops==0.8.0',
        'timm==1.0.9',
        'opencv-python-headless==4.10.0.84',
    ],
    run_commands=[
        "mkdir -p /root/.cache/torch/hub/checkpoints/florence-2/",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/pytorch_model.bin?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/pytorch_model.bin",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/config.json?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/config.json",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/configuration_florence2.py?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/configuration_florence2.py",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/generation_config.json?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/generation_config.json",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/modeling_florence2.py?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/modeling_florence2.py",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/preprocessor_config.json?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/preprocessor_config.json",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/processing_florence2.py?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/processing_florence2.py",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/tokenizer.json?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/tokenizer.json",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/tokenizer_config.json?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/tokenizer_config.json",
        "wget -q https://huggingface.co/microsoft/Florence-2-large/resolve/main/vocab.json?download=true -O /root/.cache/torch/hub/checkpoints/florence-2/vocab.json",
    ],
    system_packages=[
        "ffmpeg",
    ],
    cuda_version="12.3",
    gpu=sieve.gpu.L4(),
    restart_on_error=False,
    metadata=metadata,
)
class Florence2Model:
    def __setup__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM 

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained("/root/.cache/torch/hub/checkpoints/florence-2/", trust_remote_code=True, torch_dtype=self.torch_dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained("/root/.cache/torch/hub/checkpoints/florence-2/", trust_remote_code=True)

        warmup_input = self.processor(
            text="<OD>", 
            images=torch.zeros((1, 3, 224, 224), dtype=torch.uint8)
        ).to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=warmup_input["input_ids"].to(self.device),
            pixel_values=warmup_input["pixel_values"].to(self.device),
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        self.processor.post_process_generation(generated_text, task="<OD>", image_size=(224, 224))

    def __predict__(
            self, 
            file: sieve.File, 
            task_prompt: Literal[
                "<OD>",
                "<CAPTION_TO_PHRASE_GROUNDING>",
                "<CAPTION>",
                "<DETAILED_CAPTION>",
                "<MORE_DETAILED_CAPTION>",
                "<DENSE_REGION_CAPTION>",
                "<REGION_PROPOSAL>",
                "<OCR>",
                "<OCR_WITH_BOXES>",
                "<REGION_TO_SEGMENTATION>",
                "<REGION_TO_CATEGORY>",
                "<REGION_TO_DESCRIPTION>",
                "<REFERRING_EXPRESSION_SEGMENTATION>",
                "<OPEN_VOCABULARY_DETECTION>",
            ] = "<OD>",
            text_input: str = "", 
            debug_visualization: bool = True,
            start_frame: int = -1,
            end_frame: int = -1,
            frame_interval: int = 1,
        ):
        """
        :param file: Input image or video file.
        :param task_prompt: Task prompt to guide the model's prediction. For more information, refer to the README.
        :param text_input: Additional text input to refine the task prompt for certain tasks. Only works for certain tasks, such as <CAPTION_TO_PHRASE_GROUNDING>. For more information, refer to the README.
        :param debug_visualization: If True, returns a visualized image or video with bounding boxes and labels. Otherwise, returns parsed model output. Only works for object detection tasks.
        :param start_frame: Start frame for video processing. If -1, the video will be processed from the beginning.
        :param end_frame: End frame for video processing. If -1, the video will be processed until the end.
        :param frame_interval: Interval between frames to process. If 1, all frames will be processed. If 2, every other frame will be processed, and so on. Used to speed up video processing.
        :return: If debug_visualization is True, returns a tuple of the visualized image or video file and parsed model output as a dict or list of dicts. Otherwise, returns parsed model output as a dict for an image or list of dicts for video.
        """
        import os
        import cv2

        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        video_exts = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"]

        file_path = file.path
        file_ext = os.path.splitext(file_path)[1].lower()
        is_video = file_ext in video_exts
        is_image = file_ext in image_exts

        if not is_video and not is_image:
            raise ValueError("Unsupported file format. Supported formats are: " + ", ".join(image_exts + video_exts))
        
        if frame_interval < 1:
            raise ValueError("frame_interval must be greater than or equal to 1.")

        if end_frame > 0 and end_frame <= start_frame:
            raise ValueError("end_frame must be greater than start_frame.")

        if is_video:
            print("Processing video...")
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise Exception("Failed to open video file.")

            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) / frame_interval

            if end_frame > 0:
                end_frame = min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            if debug_visualization:
                tmp_out_path = "output.mp4"
                if os.path.exists(tmp_out_path):
                    os.remove(tmp_out_path)
                out_vid_writer = cv2.VideoWriter(tmp_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            frame_count = 0
            approx_frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            parsed_answer_list = []
            while True:
                if frame_count // frame_interval % 100 == 0:
                    print(f"Processing frame {frame_count}/{approx_frame_total}...")
                ret, frame = cap.read()
                if not ret or (end_frame > 0 and frame_count > end_frame):
                    break

                out = self.process_image(frame, task_prompt, text_input, debug_visualization)
                if debug_visualization:
                    out_image, parsed_answer = out
                    out_vid_writer.write(out_image)
                else:
                    parsed_answer = out

                parsed_answer["frame_number"] = frame_count

                parsed_answer_list.append(parsed_answer)

                frame_count += 1

                for _ in range(frame_interval - 1):
                    ret, _ = cap.read()
                    frame_count += 1
                    if not ret or (end_frame > 0 and frame_count > end_frame):
                        break 
                

            cap.release()
                

            if debug_visualization:
                out_vid_writer.release()
                import subprocess

                if os.path.exists("final_output.mp4"):
                    os.remove("final_output.mp4")

                print("Re-encoding the video with the original audio track...")

                # Use ffmpeg to re-encode the video and add the original audio track
                command = [
                    "ffmpeg",
                    "-loglevel", "error", # Suppress ffmpeg logs
                    "-i", tmp_out_path,  # Input the processed video without audio
                    "-i", file_path,  # Input the original video with audio
                    "-y",  # Overwrite the output file if it already exists
                    "-c:v", "libx264",  # Re-encode the video using H.264 codec
                    "-map", "0:v:0",  # Map the video from the processed video
                    "-map", "1:a:0?",  # Map the audio from the original video
                    "final_output.mp4"  # Output file
                ]

                subprocess.run(command, check=True)

                print("Video processing complete.")

                return sieve.File("final_output.mp4"), parsed_answer_list
                
            return parsed_answer_list
        
        # If the input is an image, process it directly
        print("Processing image...")
        image = cv2.imread(file.path)

        out = self.process_image(image, task_prompt, text_input, debug_visualization)

        if debug_visualization:
            import cv2
            out_image, parsed_answer = out
            cv2.imwrite("output.jpg", out_image)
            return sieve.File("output.jpg"), parsed_answer
        
        return out
    

    def process_image(self, image, task_prompt, text_input, debug_visualization):
        
        from PIL import Image

        pil_image = Image.fromarray(image).convert("RGB")

        prompt_input = task_prompt + text_input

        inputs = self.processor(text=prompt_input, images=pil_image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_image.width, pil_image.height))

        if debug_visualization:
            import cv2 
            out_image = image.copy()

            try:

                # Draw bounding boxes and labels on the image
                for bbox, label in zip(parsed_answer[task_prompt]['bboxes'], parsed_answer[task_prompt]['labels']):
                    # Unpack the bounding box coordinates
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    
                    # Draw the rectangle on the image
                    cv2.rectangle(out_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
                    
                    # Put the label text above the rectangle
                    cv2.putText(out_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                return out_image, parsed_answer
            
            except Exception as e:
                print("Rendering failed. Returning parsed answer only.")
                print(e)
                return parsed_answer

        return parsed_answer
    
if __name__ == "__main__":
    import cv2
    import numpy as np

    out = Florence2Model()(sieve.File("https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/c4d968f5-f25a-412b-9102-5b6ab6dafcb4/bbaafbea-1193-4ab3-97e8-2fa3c7901ae8-juggling_pins.mp4"), debug_visualization=True)

    print(out)
    