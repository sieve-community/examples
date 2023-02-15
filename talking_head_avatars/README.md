# Talking Head Avatars

Given an input video, input audio, and input avatar image, this workflow makes the avatar image into a talking person that looks like they're saying the audio with the visual motions of the person in the video.
- Detect faces and track
- Use [Wav2lip](https://github.com/Rudrabha/Wav2Lip) to lipsync a driving video
- Use [this](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model) talking head model to take the lipsync as input and generate an avatar

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

## Example Videos

| Avatar Image | Talking Avatar |
| --- | --- |
| ![stock-man-whi](https://user-images.githubusercontent.com/11367688/219148160-155afbae-f611-4998-b414-6e80d0fc4ed2.jpg) | <video src="https://user-images.githubusercontent.com/11367688/219148404-285fdab3-6926-4dc8-8af3-055f6b57c76c.mp4">
| ![stock-woman-asian](https://user-images.githubusercontent.com/11367688/219148385-8db01e8f-9c14-4b5b-8681-eb6f54fda334.jpg) | <video src="https://user-images.githubusercontent.com/11367688/219148409-a80cb815-c34c-40ca-92d0-ed5a4ee5605f.mp4">
| ![Bob_Iger_hi](https://user-images.githubusercontent.com/11367688/219144501-479ad9e9-5264-4d78-a05b-7dcca9bc538f.jpg) | <video src="https://user-images.githubusercontent.com/11367688/219144374-93f482c6-0d17-4841-a85b-a30821299741.mp4">










