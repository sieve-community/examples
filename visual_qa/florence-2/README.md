# Florence-2

This is Sieve's open-source implementation of Florence-2, a visual foundation model for a variety of image question-and-answer tasks. 

This implementation is based on the [HuggingFace implementation](https://huggingface.co/microsoft/Florence-2-large) of Florence-2 Large using the transformers library. You can find the original paper [here](https://arxiv.org/pdf/2311.06242).

For examples, click [here](#examples).

## Features
- **Object Detection**: Detect objects with an optional guiding text prompt.
- **Captioning**: Caption images with varying levels of detail.
- **OCR**: Optical character recognition to understand text in image.
- **Object Segmentation**: Text promptable segmentation of objects in image.

## Pricing
This function is hosted on an L4 GPU and is billed at a compute-based pay-as-you rate of $1.25/hr. You can find more information about compute pricing on Sieve [here](https://www.sievedata.com/pricing).

## Parameters

- `image`: A sieve.File pointing to an image to perform QA tasks.
- `task_prompt`: A string that decides what task Florence-2 should perform. For more information on the options, click [here](#task-prompt).
- `text_input`: An optional string that supplies an additional text prompt. This is only applicable for certain tasks, including `<CAPTION_TO_PHRASE_GROUNDING>`, `<REFERRING_EXPRESSION_SEGMENTATION>`, `<OPEN_VOCABULARY_DETECTION>`, `<REGION_TO_CATEGORY>`, `<REGION_TO_SEGMENTATION>`, and `<REGION_TO_DESCRIPTION>`. Other tasks will throw an error. For more info on proper usage, click [here](#text-input)
- `debug_visualization`: A boolean flag that, when set to true, enables the visualization of outputs on the source image for debugging purposes. Only works for tasks that output bounding boxes.

## Notes

### Output Format

For all tasks, the output is a dictionary where the key is the `task_prompt`. Depending on the task, the value's type changes.

For `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OCR>` and `<DENSE_REGION_CAPTION>` the value is a string.

For `<OD>` and `<CAPTION_TO_PHRASE_GROUNDING>`, the value is a dict with two keys, `bboxes` and `labels`, that point to a list of x1,y1,x2,y2 boxes and their labels, respectively.

For more information and examples, refer [here](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb).

### Task Prompt

Options for the `task_prompt` param include:
- `<OD>`: Object Detection
- `<CAPTION_TO_PHRASE_GROUNDING>`: Caption to Phrase Grounding
- `<CAPTION>`: Caption Generation
- `<DETAILED_CAPTION>`: Detailed Caption Generation
- `<MORE_DETAILED_CAPTION>`: More Detailed Caption Generation
- `<DENSE_REGION_CAPTION>`: Dense Region Captioning
- `<REGION_PROPOSAL>`: Region Proposal
- `<OCR>`: Optical Character Recognition
- `<OCR_WITH_BOXES>`: OCR with Bounding Boxes
- `<REGION_TO_SEGMENTATION>`: Segment a Specific Region
- `<REGION_TO_CATEGORY>`: Categorize a Specific Region with a One Word Descriptor
- `<REGION_TO_DESCRIPTION>`: Describe a Specific Region
- `<REFERRING_EXPRESSION_SEGMENTATION>`: Caption to Generating a Segment
- `<OPEN_VOCABULARY_DETECTION>`: OCR and Object Detection

For more information, refer [here](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb).

### Text Input

For certain tasks, like `<CAPTION_TO_PHRASE_GROUNDING>`, you can supply a prompt to the `text_input` parameter to focus on detecting/segmenting particular objects. Good phrases are 1-2 words long, as adjectives can be picked up as objects, nouns, and verbs.

For more information, refer [here](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb).

## Examples

![Example Image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true)

Let's apply Sieve's Florence-2 to this photo of a car! Be sure to log in via `sieve login` on your terminal or by setting the `SIEVE_API_KEY` environment variable.

### Object Detection

For object detection, use a code snippet like this. 

```python

import sieve

fl2_fn = sieve.function.get("sieve/florence-2")

debug_image, response = fl2_fn.run(
    image=sieve.File("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"),
    task_prompt="<OD>",
    debug_visualization=True,
)

print("debug image path:", debug_image.path)
print("dict response:", response)
```

If you run on the image of the car, the response dictionary will look something like:

```json
{
  "<OD>": {
    "bboxes": [
      [...],  // abbreviated for brevity
      [...],
      [...],
      [...]
    ],
    "labels": [
      "car",
      "door",
      "wheel",
      "wheel"
    ]
  }
}
```

### Guided Object Detection with Prompts

To detect specific objects using text, use a code snippet like this.

```python

import sieve

fl2_fn = sieve.function.get("sieve/florence-2")

text_input = "car, door."

debug_image, response = fl2_fn.run(
    image=sieve.File("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"),
    task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
    text_input=text_input,
    debug_visualization=True,
)

print("debug image path:", debug_image.path)
print("dict response:", response)
```

If you run on the image of the car, the response dictionary will look something like:

```json
{
  "<CAPTION_TO_PHRASE_GROUNDING>": {
    "bboxes": [
      [...],  // abbreviated for brevity
      [...]
    ],
    "labels": [
      "car",
      "door"
    ]
  }
}

```

The prompt matters a lot here, so we encourage you to experiment and see what works.

### Captioning

To generate a detailed caption of the image, use a code snippet like this.

```python

import sieve

fl2_fn = sieve.function.get("sieve/florence-2")

response = fl2_fn.run(
    image=sieve.File("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"),
    task_prompt="<DETAILED_CAPTION>",
)

print("dict response:", response)
```

If you run on the image of the car, the response dictionary will look something like:

```json
{
  "<DETAILED_CAPTION>": "The image shows a blue Volkswagen Beetle parked in front of a yellow building with two brown doors, surrounded by trees and a clear blue sky."
}
```

