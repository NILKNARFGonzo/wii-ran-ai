# Wii Ran AI

Wii Ran AI is a local C++ text generator for the Wii with a model picker:

- **Stories:** [StoryBananaMind-7.5M](https://huggingface.co/BananaMind/StoryBananaMind-7.5M),
  a 7.5M-parameter GPT-2 story model.
- **Chat:** [BananaMind 2 Nano Chat](https://huggingface.co/BananaMind/BananaMind-2-Nano-Chat),
  an experimental 9,968,128-parameter, full-parameter supervised fine-tune of
  BananaMind 2 Nano on Smol-SmolTalk. Its upstream card describes a 10-layer
  decoder with 256-wide hidden states, four query heads / two KV heads, QK RMS
  normalization, RoPE, SwiGLU, an 8,192-token digit-aware BPE tokenizer, and
  4,096-token context. The weights and tokenizer are Apache-2.0.
- **Larger chat:** [BananaMind 2 Medium Chat](https://huggingface.co/BananaMind/BananaMind-2-Medium-Chat),
  an experimental 49,559,552-parameter instruction-tuned model with 12 layers,
  512-wide hidden states, and a 12,288-token tokenizer. The weights and
  tokenizer are Apache-2.0.

The Wii build quantizes each model's non-normalization tensors to INT8 while
keeping normalization weights in float32; activations and KV cache are
float32. Story context stays at 256 tokens; Nano Chat context is capped at
2,048 tokens (20 MiB maximum KV cache), while Medium Chat uses a 512-token
context (6 MiB maximum KV cache), with 64 tokens reserved for replies. Their
official contexts are not exposed in full. The `.dol` embeds StoryBananaMind
and Nano Chat; Medium Chat's quantized weights and tokenizer are included as
separate SD-card files because of their size. No network connection is needed
at runtime. The chat checkpoints are pinned to upstream revisions
`6def7ab4d1bd8449d9b75b62bfb174f73d3a8ad6` (Nano) and
`86bfcd9075e54a7da8aa101d9ba2ca819fa59143` (Medium).

## Use on Wii

1. Download [`wii-ran-ai-hbc.zip`](https://github.com/NILKNARFGonzo/wii-ran-ai/releases/latest)
   from Releases and extract it to the root of an SD card. It installs as
   `apps/wii-ran-ai/boot.dol`.
2. Launch **Wii Ran AI** from the Homebrew Channel.
3. Use the D-pad and **A** to select a model. Choose Stories for prompt
   continuation, Nano Chat for the smaller assistant, or Medium Chat for the
   larger assistant.
4. Type with a USB keyboard, or use the Wii Remote D-pad to select on-screen
   letters, numbers, and punctuation. Press **A** to type, **B** to delete,
   **Minus** for a space, and **Plus** to generate a story or chat reply.
5. Press **A** or **Plus** after a response to enter another prompt/message.
   The chat model retains up to 32 recent turns, trimming older context to fit
   the selected model's Wii memory budget. Press **Home** to quit.

Inputs are limited to 256 characters. Story generation is limited to 80 tokens
or the story model's 256-token context; chat replies are limited to 64 tokens
within the 2,048-token Wii chat context. Up to 32 recent turns are retained,
trimmed as needed to fit the context. BananaMind Nano Chat is experimental and
not reliable for general assistance: its model card reports 35 passed examples
out of 300 (9.15%) on its self-reported instruction benchmark and warns that it
may fail arithmetic, hallucinate facts, or produce invalid output. The Wii app
shows this warning before chat. Do not rely on it for correctness or high-stakes
use.
Medium Chat is also experimental; its context is 512 tokens and replies are
limited to 64 tokens. Its approximately 50 MB packed model file must remain at
`apps/wii-ran-ai/medium-chat-wii-weights.bin` on the SD card, alongside
`medium-chat-wii-tokenizer.bin`. Medium Chat lowercases ASCII user input before
tokenization to accommodate the Wii Remote keyboard's uppercase-only keys.

## Build from source

Install Python 3, PyTorch, NumPy, `safetensors`, devkitPro with the Wii/
devkitPPC and libogc components, plus the `libfat-ogc` package, then run:

```sh
python3 -m pip install torch numpy safetensors
python3 download_model.py
python3 download_chat_model.py
python3 download_medium_chat_model.py
make
```

`make` downloads the pinned StoryBananaMind, Nano Chat, and Medium Chat
checkpoints, converts them to the Wii runtime format, builds the Wii DOL, and
creates `wii-ran-ai-hbc.zip`. Extract that ZIP to the root of an SD card; it
includes the separate Medium Chat model files. The archive also includes the
Apache-2.0 license text. All three checkpoints are Apache-2.0.

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
You are a concise and helpful assistant.
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
