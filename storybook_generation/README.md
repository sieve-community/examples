# Storybook Generation
This workflow generates a video story from a paragraph of text. It uses `StableDiffusionWalker` on pairwise sentences to generate video clips, captions them with each sentence, and stitches them together into a video.

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
