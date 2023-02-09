# Face Detection

Detects faces using the MediaPipe face recognition module.

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
cd examples/face_detection
sieve deploy
```

## Example Detection

![sample image](https://www.incimages.com/uploaded_files/image/1920x1080/getty_481292845_77896.jpg)

```
[
  {
    "box": [
      0.5071465969085693,
      0.18284249305725098,
      0.15523338317871094,
      0.2759702801704407
    ],
    "class_name": "face",
    "score": 0.7970966100692749
  }
]
```
