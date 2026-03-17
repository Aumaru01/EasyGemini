# EasyGemini

The simplest way to call the Google Gemini API in Python — with built-in multi-key rotation and automatic rate-limit handling.

> One function call. Multiple API keys. Zero rate-limit headaches.

## Why EasyGemini?

Calling the Gemini API should be simple. But when you're working with free-tier keys, you constantly hit rate limits — RPM, RPD, TPM — and your app crashes. EasyGemini solves this by letting you pool multiple API keys and automatically switching between them. You just call one function and get your result.

```python
from Gemini_api import gemini_api

text, prompt_tokens, output_tokens, total_tokens, key_index = gemini_api(
    user_prompt="Explain quantum computing in simple terms.",
    model_name="gemini-2.5-flash",
    config={"temperature": 0.7},
    system_instruction="You are a helpful assistant.",
)

print(text)
```

That's it. No key management. No retry logic. No rate-limit handling. It just works.

## Features

- **Dead simple API** — One function call is all you need. Pass your prompt, get your response.
- **Auto key rotation** — Round-robin across all your API keys. Maximizes throughput without any extra code.
- **Smart rate-limit handling** — Automatically detects RPM, RPD, and TPM quota errors, disables the exhausted key, and seamlessly switches to the next available one.
- **Thread-safe** — Use it in multi-threaded applications. Per-key semaphores and global locking keep everything safe.
- **Auto-recovery** — Disabled keys come back online after their cooldown expires. Daily counters reset automatically.
- **Token pre-check** — Validates input token count against model limits before sending, so you get a clear error instead of a wasted API call.
- **Multi-model support** — Works with Gemini 2.5 Flash, Gemini 3.1 Flash Lite, Gemma 3 27B, and more.
- **Debug mode** — Call `show_key_state()` to see real-time status of all your keys.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure your API keys

Copy the example config and add your keys:

```bash
cp API_Keychain.config.example API_Keychain.config
```

Then edit `API_Keychain.config` with your real API keys:

```ini
[API_KEY]
Gemini_API_KEYS = ["your-key-1", "your-key-2", "your-key-3"]
```

Get your free API key at: https://aistudio.google.com/apikey

The more keys you add, the higher your effective throughput.

> **Note:** You can also use a custom config path by creating a `path.config` file:
> ```ini
> [Path]
> API_key_chain_config = path/to/your/custom_keys.config
> ```

### 3. Use it

```python
from Gemini_api import gemini_api

text, prompt_tokens, output_tokens, total_tokens, key_index = gemini_api(
    user_prompt="Hello!",
    model_name="gemini-2.5-flash",
    config={"temperature": 0.7},
    system_instruction="You are a helpful assistant.",
)
```

**Returns:** `(response_text, prompt_tokens, output_tokens, total_tokens, key_index)`

## How It Works

1. You call `gemini_api()` with your prompt.
2. The module picks the next available key using round-robin.
3. If that key hits a rate limit, it's temporarily disabled and the next key is tried automatically.
4. Once a key's cooldown expires, it's back in the rotation.
5. If all keys are exhausted for the day, a clear `RuntimeError` is raised.

You don't need to manage any of this — it all happens behind the scenes.

## Supported Models

| Model | Max RPD (per key) | Max Input Tokens |
|---|---|---|
| `gemini-2.5-flash` | 20 | 1,000,000 |
| `gemini-2.5-flash-lite` | 20 | 1,000,000 |
| `gemini-3.1-flash-lite` | 500 | 1,000,000 |
| `gemma-3-27b-it` | 14,400 | 128,000 |

## Configuration

All constants can be tuned at the top of `Gemini_api.py`:

| Constant | Default | Description |
|---|---|---|
| `KEY_CONCURRENCY` | 10 | Max concurrent requests per key |
| `KEY_INTERVAL` | 6s | Minimum seconds between calls on the same key |
| `MAX_RETRY_WAIT` | 60s | Cooldown for unknown errors or timeouts |
| `RPM_COOLDOWN` | 60s | Cooldown after hitting requests-per-minute limit |
| `TPM_COOLDOWN` | 60s | Cooldown after hitting tokens-per-minute limit |
| `SERVER_BUSY_COOLDOWN` | 30s | Cooldown after 503 server overload |

## Debugging

```python
from Gemini_api import show_key_state

for key in show_key_state():
    print(key)
```

Output shows each key's index, active requests, daily usage, disabled status, and cooldown timer.