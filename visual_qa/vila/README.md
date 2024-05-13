# VILA: On Pre-training for Visual Language Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](CODE_LICENSE)
[![Model License](https://img.shields.io/badge/MODEL%20License-CC%20By%20NC%204.0-red.svg)](MODEL_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)


[VILA arxiv](https://arxiv.org/abs/2312.07533) / [VILA Demo](https://vila-demo.hanlab.ai/) / [VILA Huggingface](https://huggingface.co/collections/Efficient-Large-Model/vila-on-pre-training-for-visual-language-models-65d8022a3a52cd9bcd62698e)

## 💡 Introduction
VILA is a visual language model (VLM) pretrained with interleaved image-text data at scale, enabling **video understanding** and **multi-image understanding** capabilities. VILA is deployable on the edge by [AWQ](https://arxiv.org/pdf/2306.00978.pdf) 4bit quantization and [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) framework. We find: (1) image-text pairs are not enough, interleaved image-text is essential; (2) unfreezing LLM during interleaved image-text pre-training enables in-context learning; (3)re-blending text-only instruction data is crucial to boost both VLM and text-only performance; (4) token compression extends #video frames. VILA unveils appealing capabilities, including: video reasoning, in-context learning, visual chain-of-thought, and better world knowledge. 

See [this repository](https://github.com/Efficient-Large-Model/VILA) for more details on VILA and its applications.

