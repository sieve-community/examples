# Background Removal

Performs background removal on video using a small version of [u2net](https://github.com/aquadzn/background-removal).

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
A fine man dancing weirdly.

https://user-images.githubusercontent.com/11367688/217718019-386abab8-7ae6-4cc3-a6a0-a33ff76678f9.mp4

https://user-images.githubusercontent.com/11367688/217718029-ca2072db-2681-4fb5-9595-ffcaa5b502b5.mp4


