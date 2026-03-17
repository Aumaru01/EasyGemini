"""
EasyGemini — Example Usage
==========================
This file demonstrates how to use gemini_api() in various scenarios.
Make sure you have configured path.config and your API key config file
before running these examples.
"""

from Gemini_api import gemini_api, show_key_state


# ==================================================
# Example 1: Basic — Simple text generation
# ==================================================

def basic_call():
    """The simplest way to call Gemini. One function, one result."""

    text, prompt_tokens, output_tokens, total_tokens, key_index = gemini_api(
        user_prompt="What is Python?",
        model_name="gemini-2.5-flash",
        config={"temperature": 0.7},
        system_instruction="You are a helpful assistant. Answer concisely.",
    )

    print("=== Basic Call ===")
    print(f"Response : {text[:200]}...")
    print(f"Tokens   : {prompt_tokens} in / {output_tokens} out / {total_tokens} total")
    print(f"Key used : #{key_index}")
    print()


# ==================================================
# Example 2: Custom config — Control output behavior
# ==================================================

def custom_config():
    """Customize temperature, max tokens, top_p, top_k, etc."""

    config = {
        "temperature": 0.2,       # Lower = more deterministic
        "max_output_tokens": 256, # Limit response length
        "top_p": 0.9,
        "top_k": 40,
    }

    text, *_ = gemini_api(
        user_prompt="List 5 tips for writing clean Python code.",
        model_name="gemini-2.5-flash",
        config=config,
        system_instruction="You are a senior Python developer.",
    )

    print("=== Custom Config ===")
    print(text)
    print()


# ==================================================
# Example 3: Different models
# ==================================================

def different_models():
    """Switch models by changing the model_name parameter."""

    models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        # "gemini-3.1-flash-lite",
        # "gemma-3-27b-it",
    ]

    for model in models:
        text, prompt_tokens, output_tokens, total_tokens, key_index = gemini_api(
            user_prompt="Say hello in 10 words or less.",
            model_name=model,
            config={"temperature": 0.5, "max_output_tokens": 50},
            system_instruction="Be brief.",
        )

        print(f"=== {model} ===")
        print(f"Response : {text.strip()}")
        print(f"Tokens   : {total_tokens} total | Key #{key_index}")
        print()


# ==================================================
# Example 4: JSON output — Structured responses
# ==================================================

def json_output():
    """Ask Gemini to return structured JSON data."""

    text, *_ = gemini_api(
        user_prompt="Give me 3 popular programming languages with their year of creation. Return as JSON array.",
        model_name="gemini-2.5-flash",
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
        system_instruction="You are a data provider. Always respond in valid JSON.",
    )

    print("=== JSON Output ===")
    print(text)
    print()


# ==================================================
# Example 5: Safety settings
# ==================================================

def with_safety_settings():
    """Customize safety thresholds for content filtering."""

    safety = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
    }

    text, *_ = gemini_api(
        user_prompt="Explain how firewalls protect computer networks.",
        model_name="gemini-2.5-flash",
        config={"temperature": 0.5},
        system_instruction="You are a cybersecurity expert.",
        safety_settings=safety,
    )

    print("=== With Safety Settings ===")
    print(text[:300])
    print()


# ==================================================
# Example 6: Multi-threaded — Concurrent requests
# ==================================================

def multithreaded_calls():
    """Send multiple requests in parallel. Keys are rotated automatically."""

    import threading

    results = {}

    def worker(task_id, prompt):
        try:
            text, _, _, total_tokens, key_index = gemini_api(
                user_prompt=prompt,
                model_name="gemini-2.5-flash",
                config={"temperature": 0.7, "max_output_tokens": 100},
                system_instruction="Answer in one sentence.",
            )
            results[task_id] = {
                "text": text.strip(),
                "tokens": total_tokens,
                "key": key_index,
            }
        except Exception as e:
            results[task_id] = {"error": str(e)}

    prompts = {
        1: "What is the capital of Japan?",
        2: "What is the speed of light?",
        3: "Who painted the Mona Lisa?",
        4: "What is the boiling point of water?",
        5: "What is the largest ocean?",
    }

    threads = []
    for task_id, prompt in prompts.items():
        t = threading.Thread(target=worker, args=(task_id, prompt))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("=== Multi-threaded Results ===")
    for task_id in sorted(results):
        r = results[task_id]
        if "error" in r:
            print(f"  Task {task_id}: ERROR — {r['error']}")
        else:
            print(f"  Task {task_id}: Key #{r['key']} | {r['tokens']} tokens | {r['text'][:80]}")
    print()


# ==================================================
# Example 7: Error handling
# ==================================================

def with_error_handling():
    """Gracefully handle errors like token limits and key exhaustion."""

    try:
        text, *_ = gemini_api(
            user_prompt="Hello!",
            model_name="gemini-2.5-flash",
            config={"temperature": 0.7},
            system_instruction="You are a helpful assistant.",
        )
        print("=== Error Handling ===")
        print(f"Success: {text.strip()}")

    except ValueError as e:
        # Input tokens exceed model limit
        print(f"Token error: {e}")

    except RuntimeError as e:
        # All API keys exhausted / RPD-exhausted
        print(f"Key exhaustion: {e}")

    except Exception as e:
        # Safety block, invalid argument, etc.
        print(f"Other error: {e}")

    print()


# ==================================================
# Example 8: Debug — Monitor key status
# ==================================================

def debug_keys():
    """Check the real-time status of all API keys."""

    print("=== Key Status ===")
    for key in show_key_state():
        status = "DISABLED" if key["disabled"] else "ACTIVE"
        print(
            f"  Key #{key['index']}: {status} | "
            f"In-use: {key['in_use']} | "
            f"Daily: {key['daily_count']} | "
            f"Reason: {key['reason']} | "
            f"Until: {key['disabled_until']}"
        )
    print()


# ==================================================
# RUN EXAMPLES
# ==================================================

if __name__ == "__main__":
    # Uncomment the examples you want to run:

    basic_call()
    # custom_config()
    # different_models()
    # json_output()
    # with_safety_settings()
    # multithreaded_calls()
    # with_error_handling()
    # debug_keys()
