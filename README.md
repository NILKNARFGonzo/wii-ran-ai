# Wii Ran AI

Wii Ran AI is a local C++ text generator for the Wii with a model picker:

- **Stories:** [StoryBananaMind-7.5M](https://huggingface.co/BananaMind/StoryBananaMind-7.5M),
  a 7.5M-parameter GPT-2 story model.
- **Chat:** [BananaMind 2 Nano Chat](https://huggingface.co/BananaMind/BananaMind-2-Nano-Chat),
  a 9,968,128-parameter, full-parameter supervised fine-tune of BananaMind 2
  Nano on Smol-SmolTalk. Its upstream card describes a 10-layer decoder with
  256-wide hidden states, four query heads / two KV heads, QK RMS normalization,
  RoPE, SwiGLU, an 8,192-token digit-aware BPE tokenizer, and 4,096-token
  context. The weights and tokenizer are Apache-2.0.

The Wii build quantizes each model's non-normalization tensors to INT8 while
keeping normalization weights in float32; activations and KV cache are
float32. Story context stays at 256 tokens; Chat context is deliberately capped
at 384 tokens (64 tokens reserved for a reply) to bound cache memory on Wii.
The Nano Chat's official 4,096-token context is therefore not exposed in this
build. The `.dol` embeds both models and their tokenizers, so no SD model files
or network connection are needed at runtime. The chat checkpoint is pinned to
upstream revision `6def7ab4d1bd8449d9b75b62bfb174f73d3a8ad6`.

## Use on Wii

1. Download [`wii-ran-ai-hbc.zip`](https://github.com/NILKNARFGonzo/wii-ran-ai/releases/latest)
   from Releases and extract it to the root of an SD card. It installs as
   `apps/wii-ran-ai/boot.dol`.
2. Launch **Wii Ran AI** from the Homebrew Channel.
3. Use the D-pad and **A** to select a model. Choose Stories for prompt
   continuation, or Chat for question-and-answer conversation.
4. Type with a USB keyboard, or use the Wii Remote D-pad to select on-screen
   letters, numbers, and punctuation. Press **A** to type, **B** to delete,
   **Minus** for a space, and **Plus** to generate a story or chat reply.
5. Press **A** or **Plus** after a response to enter another prompt/message.
   The chat model retains up to eight recent turns, trimming older context to
   fit its 384-token Wii memory budget. Press **Home** to quit.

Inputs are limited to 256 characters. Story generation is limited to 80 tokens
or the story model's 256-token context; chat replies are limited to 64 tokens
within the 384-token Wii chat context. BananaMind Nano Chat is a research-scale
model and may give weak, incorrect, or incoherent answers; it is not suitable
for high-stakes use.

## Build from source

Install Python 3, PyTorch, NumPy, `safetensors`, devkitPro with the Wii/
devkitPPC and libogc components, then run:

```sh
python3 -m pip install torch numpy safetensors
python3 download_model.py
python3 download_chat_model.py
make
```

`make` downloads the pinned StoryBananaMind and BananaMind Nano Chat
checkpoints, converts them to the Wii runtime format, builds the Wii DOL, and
creates `wii-ran-ai-hbc.zip`. Extract that ZIP to the root of an SD card.
The archive also includes the Apache-2.0 license text. StoryBananaMind's
checkpoint and the Nano Chat checkpoint are Apache-2.0.

## Test on a computer

```sh
g++ -O3 -std=c++11 wii_story.cpp -o wii_story_host
./wii_story_host story-wii-weights.bin story-wii-tokenizer.bin prompt.txt 32
```

Arguments are weights path, tokenizer path, prompt text-file path, and the
maximum number of new tokens. The default prompt is used if the prompt file
does not exist.

For a Nano Chat host smoke test, write a model-native prompt to a text file:

```text
<|bos|><|system|>
You are BananaMind, a friendly and concise assistant. Answer directly, be honest when you are unsure, and keep explanations clear.
<|user|>
Say hello in one short sentence.<|assistant|>
```

Then run:

```sh
./wii_story_host chat-wii-weights.bin chat-wii-tokenizer.bin chat-prompt.txt 32
```

## Build for Wii

Story generation uses top-k sampling with a repetition penalty and repeated
trigram blocking. Chat uses deterministic decoding with the upstream 1.1
repetition penalty and BananaMind role markers,
special token IDs, digit-isolated byte-level BPE, QK RMSNorm, rotary position
embeddings, grouped-query attention, and SwiGLU blocks. The tokenizer's
pre-tokenization is a compact C++ implementation of the upstream digit-aware
byte-level rules. Generation is CPU-only and may be slow on Wii hardware.
This project has not been validated on physical Wii hardware.
