# Hyperfast Word-level Audio Transcription

This app is able to transcribe over 40 mins of audio in ~30 seconds using the state-of-the-art audio transcription model Whisper Large-V2. It does this by first detecting silent sections in an audio clip and then parallelizing the transcription of the non-silent sections across many instances of the model to deliver hyper-fast transcription speeds. Try it for yourself.

It returns the transcription in a JSON format as shown below:

```json
[
  {
    "start": 633.4218800000001,
    "end": 635.7838800000001,
    "text": " system.",
    "words": [
      {
        "start": 633.4218800000001,
        "end": 633.8018800000001,
        "score": 0.855,
        "word": "system."
      }
    ]
  },
  {
    "start": 635.7838800000001,
    "end": 638.48788,
    "text": "They've done things that were unthinkable.",
    "words": [
      {
        "start": 635.7838800000001,
        "end": 635.9848800000001,
        "score": 0.914,
        "word": "They've"
      },
      {
        "start": 636.02488,
        "end": 636.16488,
        "score": 0.779,
        "word": "done"
      },
      {
        "start": 636.2248800000001,
        "end": 636.44488,
        "score": 0.78,
        "word": "things"
      },
      {
        "start": 636.4648800000001,
        "end": 636.56488,
        "score": 0.767,
        "word": "that"
      },
      {
        "start": 636.6048800000001,
        "end": 636.74588,
        "score": 0.85,
        "word": "were"
      },
      {
        "start": 636.88588,
        "end": 637.5668800000001,
        "score": 0.893,
        "word": "unthinkable."
      }
    ]
  },
  {
    "start": 638.48788,
    "end": 639.8688800000001,
    "text": "I say we don't have a free press.",
    "words": [
      {
        "start": 638.48788,
        "end": 638.5678800000001,
        "score": 0.752,
        "word": "I"
      },
      {
        "start": 638.6278800000001,
        "end": 638.7878800000001,
        "score": 0.949,
        "word": "say"
      },
      {
        "start": 638.82788,
        "end": 638.9278800000001,
        "score": 0.977,
        "word": "we"
      },
      {
        "start": 638.96788,
        "end": 639.10788,
        "score": 0.906,
        "word": "don't"
      },
      {
        "start": 639.1478800000001,
        "end": 639.2488800000001,
        "score": 0.888,
        "word": "have"
      },
      {
        "start": 639.30888,
        "end": 639.34888,
        "score": 0.595,
        "word": "a"
      },
      {
        "start": 639.3888800000001,
        "end": 639.56888,
        "score": 0.768,
        "word": "free"
      },
      {
        "start": 639.6088800000001,
        "end": 639.84888,
        "score": 0.889,
        "word": "press."
      }
    ]
  }
]
```
