import sieve
from typing import Literal

metadata = sieve.Metadata(
    title="Florence-2",
    description="A visual language foundation model that can perform a variety of question-answer tasks, such as object detection, image captioning, segmentation, and OCR.",
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
            image: sieve.File, 
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
        ):
        """
        :param image: Input image file.
        :param task_prompt: Task prompt to guide the model's prediction. For more information, refer to the README.
        :param text_input: Additional text input to refine the task prompt for certain tasks. Only works for certain tasks, such as <CAPTION_TO_PHRASE_GROUNDING>. For more information, refer to the README.
        :param debug_visualization: If True, returns a visualized image with bounding boxes and labels. Otherwise, returns parsed model output. Only works for object detection tasks.
        :return: If debug_visualization is True, returns a tuple of the visualized image file and parsed model output as a dict. Otherwise, returns parsed model output as a dict.
        """

        from PIL import Image

        pil_image = Image.open(image.path).convert("RGB")

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
            temp_image_path = "output.jpg"

            out_image = cv2.imread(image.path)

            try:

                # Draw bounding boxes and labels on the image
                for bbox, label in zip(parsed_answer[task_prompt]['bboxes'], parsed_answer[task_prompt]['labels']):
                    # Unpack the bounding box coordinates
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    
                    # Draw the rectangle on the image
                    cv2.rectangle(out_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
                    
                    # Put the label text above the rectangle
                    cv2.putText(out_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                cv2.imwrite(temp_image_path, out_image)

                return sieve.File(temp_image_path), parsed_answer
            
            except Exception as e:
                print("Rendering failed. Returning parsed answer only.")
                print(e)
                return parsed_answer

        return parsed_answer