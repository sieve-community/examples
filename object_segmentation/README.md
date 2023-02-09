# Object Segmentation

Performs common object segmentation on a video, and returns the original video with the segmentation masks overlayed.

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
cd examples/object_segmentation
sieve deploy
```

## Example Video
Lebron James and Dwayne Wade having a colorful conversation.

https://user-images.githubusercontent.com/8021950/217714331-5a8ef1b0-b8e3-44d1-8f34-a35721e61b8c.mp4
