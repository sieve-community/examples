# Storybook Generation
This workflow generates a video story from a paragraph of text. It uses `StableDiffusionWalker` on pairwise sentences to generate video clips, captions them with each sentence, and stitches them together into a video.

## Examples
Here's a good starter prompt: `Once upon a time, there was a small bird named Poppy. Poppy was a curious bird who loved to explore the world around her. One day, as she was flying over the fields, she noticed a beautiful flower in the distance. Poppy flew closer to the flower and was amazed by its vibrant colors and sweet fragrance. She landed on the flower and started to sip the nectar from its center.`

## Deploying
Follow our [getting started guide](https://www.sievedata.com/dashboard/welcome) to get your Sieve API key and install the Sieve Python client.

1. Export API keys & install Python client
```
export SIEVE_API_KEY={YOUR_API_KEY}
pip install https://mango.sievedata.com/v1/client_package/sievedata-0.0.1.1.2-py3-none-any.whl
```

2. Deploy a workflow to Sieve
```
git clone git@github.com:sieve-community/examples.git
cd examples/yolo_object_tracking
sieve deploy
```
