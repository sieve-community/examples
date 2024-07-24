# Demucs Source Seperation

Demucs is a state-of-the-art music source separation model, currently capable of separating drums, bass, and vocals from the rest of the accompaniment. Demucs is based on a U-Net convolutional architecture inspired by Wave-U-Net. The v4 version features Hybrid Transformer Demucs, a hybrid spectrogram/waveform separation model using Transformers. It is based on Hybrid Demucs (also provided in this repo), with the innermost layers replaced by a cross-domain Transformer Encoder. This Transformer uses self-attention within each domain, and cross-attention across domains. The model achieves a SDR of 9.00 dB on the MUSDB HQ test set. Moreover, when using sparse attention kernels to extend its receptive field and per source fine-tuning, we achieve state-of-the-art 9.20 dB of SDR.


## Usage

To use this function, you need to call the `demucs` function with the sieve library with following parameters:

* `audio` -  Audio file (mp3,wav,flac) to be seperated
* `model` - The model to be used for audio separation. Default is "htdemucs_ft". Available models are given below
* `two_stems` - The two_stems option seperates the stem chosen from the rest of the stems. 
* `overlap` -  option controls the amount of overlap between prediction windows. Default is 0.25 (i.e. 25%) which is probably fine. It can probably be reduced to 0.1 to improve a bit speed.
* `shifts` - The number of shifts to be used. performs multiple predictions with random shifts (a.k.a the shift trick) of the input and average them. This makes prediction SHIFTS times slower. Default is 0
* `mp3` - If True, the audio will be saved as mp3. Default is False
* `Returns` - A background audio and a foreground audio.


## Available Models and details
- **`htdemucs`**: First version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
- **`htdemucs_ft`**: Fine-tuned version of `htdemucs`, separation will take 4 times more time but might be a bit better. Same training set as `htdemucs`.
- **`htdemucs_6s`**: 6 sources version of `htdemucs`, with piano and guitar being added as sources. Note that the piano source is not working great at the moment.
- **`hdemucs_mmi`**: Hybrid Demucs v3, retrained on MusDB + 800 songs.
- **`mdx`**: Trained only on MusDB HQ, winning model on track A at the MDX challenge.
- **`mdx_extra`**: Trained with extra training data (including MusDB test set), ranked 2nd on the track B of the MDX challenge.
- **`mdx_q`, `mdx_extra_q`**: Quantized version of the previous models. Smaller download and storage but quality can be slightly worse.

**Note**: If you'd like us to support any other models reach out to us!