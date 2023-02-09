# Stable Diffusion Walker

Generates a video that walks through the latent space of the stable diffusion embeddings given a start prompt and end prompt.

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
cd examples/stable_diffusion_walker
sieve deploy
```

## Example Video
Morphing from "apples in a basket" to "bananas in a basket"

https://user-images.githubusercontent.com/6136843/217713060-e03e4805-7b8d-4e0d-b43a-38f4ae77f6a6.mp4
