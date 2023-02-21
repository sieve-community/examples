# Custom Tortoise Text-to-Speech

Generates audio that sounds like the voice in the input audio using TortoiseTTS given a piece of text.

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
cd examples/custom_text_to_speech
sieve deploy
```

## Usage
1. Upload 2 samples of target voice, about 10-15 seconds each
2. Set preset to one of the following:
    - ultra_fast
    - fast
    - standard
    - high_quality
3. Enter the text you would like to be generated in the target voice

An example of using the API in a webapp can be found [here](https://github.com/gaurangbharti1/tortoise-streamlit)
## Example Generation using Morgan Freeman's voice

```
"Fourscore and seven years ago our fathers brought forth, on this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived, and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting-place for those who here gave their lives, that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we cannot dedicate, we cannot consecrate—we cannot hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they here gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom, and that government of the people, by the people, for the people, shall not perish from the earth."
```

[freeman_getty.webm](https://user-images.githubusercontent.com/11367688/217720684-b98e13c1-f04a-44bd-bf73-5d4c34d7d0e6.webm)
