# Talking Head Avatars

Given an input video, input audio, and input avatar image, this workflow makes the avatar image into a talking person that looks like they're saying the audio with the visual motions of the person in the video (TODO: explanation on how).

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
cd examples/talking_head_avatars
sieve deploy
```

## Example Video

https://user-images.githubusercontent.com/11367688/219099107-9ae8cd41-01ad-49cc-afd3-5f863e5df4f7.mp4



https://user-images.githubusercontent.com/11367688/219099121-d7a7f0a0-9ae2-4155-964a-028f77fac565.mp4

