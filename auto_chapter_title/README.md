# Auto Video Chapter Titles

Automatically generate video chapter titles with Whisper, GPT-3, and text segmentation. Read more [on our blog](https://www.sievedata.com/blog/ai-auto-video-chapters).

## Deploying
Follow our [getting started guide](https://www.sievedata.com/dashboard/welcome) to get your Sieve API key and install the Sieve Python client.

1. Export API keys, install Python client, and clone this repo
```
export SIEVE_API_KEY={YOUR_API_KEY}
pip install https://mango.sievedata.com/v1/client_package/sievedata-0.0.1.1.2-py3-none-any.whl
git clone git@github.com:sieve-community/examples.git
```

2. Add your OPENAI_API_KEY to the `auto_chapter_title/.env` file
```
OPENAI_API_KEY={YOUR_API_KEY}
```

3. Deploy to sieve
```
cd examples/auto_chapter_title
sieve deploy
```

## Example Generation

[Sample YouTube Video](https://www.youtube.com/watch?v=0gNauGdOkro)

```json
[
  {
      "name": "Introduction",
      "start": 0,
      "end": 102.32
  },
  {
      "name": "OpenAI's ChatGPT AI Chatbot",
      "start": 102.32,
      "end": 193.72
  },
  {
      "name": "The Use of AI Tools as a Creative Tool",
      "start": 193.72,
      "end": 314.59999999999997
  },
  {
      "name": "AI Art and Impacting the Art World",
      "start": 314.59999999999997,
      "end": 631.48
  },
  {
      "name": "AI Art and Copyright Infringement",
      "start": 631.48,
      "end": 932.4000000000001
  }
]
```
