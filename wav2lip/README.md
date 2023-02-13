# Video Lipsyncing

Lip syncs an input video to an input audio track using [Wav2Lip](https://github.com/sieve-community/wav2lip/tree/main/Wav2Lip).

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
cd examples/video_lipsyncing
sieve deploy
```

## Example Video

### Original Video
https://user-images.githubusercontent.com/11367688/218600458-bd808893-9980-4ed3-9c99-d1a3b38947b2.mov

### Lipsynced Video
https://user-images.githubusercontent.com/11367688/218600468-2fc3c465-d54b-4961-8e5b-ca638127ae2c.mp4

