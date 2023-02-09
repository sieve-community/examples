# Image Captioning

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
cd examples/image_captioning
sieve deploy
```

## Example Caption

![sample image](https://www.incimages.com/uploaded_files/image/1920x1080/getty_481292845_77896.jpg)
*a man sitting at a table with a laptop*
