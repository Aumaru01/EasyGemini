import ast
import time
import threading
import google.generativeai as genai
import google.generativeai.types as gtypes

from configparser import ConfigParser
from datetime import date, datetime, timedelta
from google.api_core.exceptions import ResourceExhausted, InvalidArgument, DeadlineExceeded


# ==================================================
# CONFIG
# ==================================================

KEY_CONCURRENCY = 10
KEY_INTERVAL = 6
MAX_RETRY_WAIT = 60
RPM_COOLDOWN = 60
TPM_COOLDOWN = 60
SERVER_BUSY_COOLDOWN = 30


# ==================================================
# LOAD API KEYS
# ==================================================

DEFAULT_KEYCHAIN = "API_Keychain.config"

path_parser = ConfigParser()
path_parser.read("path.config")
api_key_chain_path = (
    path_parser.get("Path", "API_key_chain_config")
    if path_parser.has_option("Path", "API_key_chain_config")
    else DEFAULT_KEYCHAIN
)


def load_api_keys(path=api_key_chain_path):
    parser = ConfigParser()
    parser.read(path)
    raw = parser.get("API_KEY", "Gemini_API_KEYS")
    return ast.literal_eval(raw)


API_KEYS = load_api_keys()


# ==================================================
# KEY STATE
# ==================================================

class GeminiKey:
    def __init__(self, key: str, index: int):
        self.key = key
        self.index = index

        self.semaphore = threading.Semaphore(KEY_CONCURRENCY)

        self.last_call = 0.0
        self.daily_count = 0
        self.day = date.today().isoformat()

        self.disabled = False
        self.disabled_until: float | None = None
        self.disable_reason = "-"


KEYS = [GeminiKey(k, i) for i, k in enumerate(API_KEYS)]
_global_lock = threading.Lock()
# genai.configure() mutates global state — gate configure+model-creation to one thread at a time.
# generate_content() is safe to call in parallel once the model object is created.
_genai_lock = threading.Lock()
_current_key_index = 0


# ==================================================
# UTIL
# ==================================================

def next_reset_time():
    tomorrow = datetime.now() + timedelta(days=1)
    return datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, 15, 0, 0
    ).timestamp()


def max_rpd(model_name: str) -> int:
    if model_name == "gemini-2.5-flash":
        return 20
    elif model_name == "gemini-2.5-flash-lite":
        return 20
    elif model_name == "gemini-3.1-flash-lite":
        return 500
    elif model_name == "gemma-3-27b-it":
        return 14400
    return 20


def max_token(model_name: str) -> int:
    if model_name == "gemini-2.5-flash":
        return 1_000_000
    elif model_name == "gemini-2.5-flash-lite":
        return 1_000_000
    elif model_name == "gemini-3.1-flash-lite":
        return 1_000_000
    elif model_name == "gemma-3-27b-it":
        return 128_000
    return 250_000


def classify_429(exc: Exception) -> str:
    msg = str(exc).lower()
    if "requests per minute" in msg or "rpm" in msg:
        return "RPM"
    if "requests per day" in msg or "rpd" in msg or "quota_value: 20" in msg:
        return "RPD"
    if "tokens per minute" in msg or "tpm" in msg:
        return "TPM"
    return "UNKNOWN"


def is_model_safety_block(msg: str) -> bool:
    msg = msg.lower()
    return (
        "finish_reason" in msg
        or "reciting from copyrighted" in msg
        or "response.text quick accessor" in msg
    )


def is_server_overload(msg: str) -> bool:
    msg = msg.lower()
    return "503" in msg and "overloaded" in msg


# ==================================================
# KEY ACQUIRE / RELEASE
# ==================================================

def _reset_all_keys_if_new_day():
    """Reset every key's daily counters and unlock RPD-disabled keys on day rollover.

    Must be called while holding _global_lock.
    Only RPD/local-RPD locks are cleared; short-term cooldowns (RPM, TPM, etc.)
    are left intact so they can expire naturally.
    """
    today = date.today().isoformat()
    for k in KEYS:
        if k.day != today:
            k.day = today
            k.daily_count = 0
            if k.disable_reason in ("RPD", "RPD(local)"):
                k.disabled = False
                k.disabled_until = None
                k.disable_reason = "-"


def _all_keys_rpd_exhausted() -> bool:
    """True when every key is disabled until at least the next daily reset."""
    now = time.time()
    reset = next_reset_time()
    return all(
        k.disabled and k.disabled_until is not None and k.disabled_until >= reset
        for k in KEYS
    )


def acquire_key(model_name: str) -> GeminiKey:
    """
    Return the next available key using round-robin selection.

    Never sleeps while holding _global_lock — keys whose RPM interval has not
    elapsed are skipped and tried on the next outer iteration.
    """
    global _current_key_index

    while True:
        with _global_lock:
            now = time.time()

            # Reset ALL keys at once on day rollover before selecting any key.
            _reset_all_keys_if_new_day()

            # Scan all keys once per lock acquisition looking for a ready one.
            for _ in range(len(KEYS)):
                key = KEYS[_current_key_index]

                # Revive key whose cooldown has expired.
                if key.disabled:
                    if key.disabled_until is not None and now >= key.disabled_until:
                        key.disabled = False
                        key.disabled_until = None
                        key.disable_reason = "-"
                    else:
                        _current_key_index = (_current_key_index + 1) % len(KEYS)
                        continue

                # Local RPD guard.
                if key.daily_count >= max_rpd(model_name):
                    key.disabled = True
                    key.disabled_until = next_reset_time()
                    key.disable_reason = "RPD(local)"
                    _current_key_index = (_current_key_index + 1) % len(KEYS)
                    continue

                # Skip key if its per-key RPM interval hasn't elapsed yet.
                # Caller will retry after a short sleep instead of blocking here.
                if now < key.last_call + KEY_INTERVAL:
                    _current_key_index = (_current_key_index + 1) % len(KEYS)
                    continue

                # Concurrency limit.
                if not key.semaphore.acquire(blocking=False):
                    _current_key_index = (_current_key_index + 1) % len(KEYS)
                    continue

                # Key is ready — claim it and advance index for the next caller.
                key.last_call = time.time()
                _current_key_index = (_current_key_index + 1) % len(KEYS)
                return key

        # No key ready right now — fast-fail if every key is RPD-exhausted.
        if _all_keys_rpd_exhausted():
            raise RuntimeError("All API keys are RPD-exhausted for today")

        time.sleep(0.1)


def release_key(key: GeminiKey):
    if key.semaphore._value < KEY_CONCURRENCY:
        key.semaphore.release()


# ==================================================
# GEMINI CALL
# ==================================================

def _call_gemini_once(
    key: GeminiKey,
    user_prompt: str,
    model_name: str,
    config: dict,
    system_instruction: str,
    safety_settings: dict,
    tools,
):
    # genai.configure() sets global state.  Create the model object while holding
    # _genai_lock so the correct API key is captured in the model's internal client.
    # generate_content() uses the client captured at construction, so it's safe to
    # call outside the lock, allowing concurrent requests from different keys.
    with _genai_lock:
        genai.configure(api_key=key.key)
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
            generation_config=config,
            safety_settings=safety_settings,
            tools=tools,
        )

    token_info = model.count_tokens(user_prompt)
    if token_info.total_tokens > max_token(model_name):
        raise ValueError("Input tokens exceed model limit")

    response = model.generate_content(
        user_prompt,
        request_options=gtypes.RequestOptions(timeout=200),
    )

    return (
        response.text,
        response.usage_metadata.prompt_token_count,
        response.usage_metadata.candidates_token_count,
        response.usage_metadata.total_token_count,
    )


# ==================================================
# PUBLIC API
# ==================================================

def gemini_api(
    user_prompt: str,
    model_name: str,
    config: dict,
    system_instruction: str,
    safety_settings: dict = {},
    tools=None,
):
    last_exc = None

    for _ in range(len(KEYS)):
        key = acquire_key(model_name)

        try:
            result = _call_gemini_once(
                key,
                user_prompt,
                model_name,
                config,
                system_instruction,
                safety_settings,
                tools,
            )

            with _global_lock:
                key.daily_count += 1
            return (*result, key.index)

        except ResourceExhausted as e:
            quota = classify_429(e)
            key.disabled = True
            key.disable_reason = quota

            if quota == "RPM":
                key.disabled_until = time.time() + RPM_COOLDOWN
            elif quota == "TPM":
                key.disabled_until = time.time() + TPM_COOLDOWN
            elif quota == "RPD":
                key.disabled_until = next_reset_time()
            else:
                key.disabled_until = time.time() + MAX_RETRY_WAIT

            last_exc = e

        except InvalidArgument:
            raise

        except DeadlineExceeded as e:
            key.disabled = True
            key.disabled_until = time.time() + MAX_RETRY_WAIT
            key.disable_reason = "TIMEOUT"
            last_exc = e

        except Exception as e:
            msg = str(e)

            if is_model_safety_block(msg):
                raise

            if is_server_overload(msg):
                key.disabled = True
                key.disabled_until = time.time() + SERVER_BUSY_COOLDOWN
                key.disable_reason = "SERVER_BUSY"
                last_exc = e
                continue

            key.disabled = True
            key.disabled_until = time.time() + MAX_RETRY_WAIT
            key.disable_reason = "UNKNOWN"
            last_exc = e

        finally:
            release_key(key)

    raise RuntimeError("All API keys exhausted") from last_exc


# ==================================================
# DEBUG
# ==================================================

def show_key_state():
    now = time.time()
    out = []

    for k in KEYS:
        if k.disabled_until is None:
            until = None
        elif now >= k.disabled_until:
            until = "READY"
        else:
            until = datetime.fromtimestamp(k.disabled_until).isoformat()

        out.append({
            "index": k.index,
            "in_use": KEY_CONCURRENCY - k.semaphore._value,
            "daily_count": k.daily_count,
            "disabled": k.disabled,
            "reason": k.disable_reason,
            "disabled_until": until,
        })

    return out
