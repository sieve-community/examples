# Audio Noise Reduction

Removes a lot of background noise and distraction from an input audio track using [FullSubNet+](https://github.com/hit-thusz-RookieCJ/FullSubNet-plus).

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
cd examples/audio_noise_reduction
sieve deploy
```

## Example Audio

### Original Audio
[noisy.webm](https://user-images.githubusercontent.com/11367688/218657325-a1287063-9c4c-4f5d-847a-98fd7c26f255.webm)

### Cleaned Audio
[fullsubnet.webm](https://user-images.githubusercontent.com/11367688/218657335-f5060714-4d22-4d73-8b6d-85d1f38c1481.webm)
