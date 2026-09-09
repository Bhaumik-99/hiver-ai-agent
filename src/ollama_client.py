"""
Thin wrapper around the Ollama HTTP API and optional Groq API for LLM-as-judge.
Supports both local Ollama and cloud Groq inference.
"""

import json
import re
import time
import requests
import os


OLLAMA_BASE = "http://localhost:11434"
GROQ_BASE = "https://api.groq.com/openai/v1"

# Which Groq model the agent uses. Overridable via GROQ_MODEL in the environment
# so a run can be moved to another model without editing code — necessary in
# practice because each model carries its own per-day token budget.
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def _ollama_generate(prompt: str, model: str = "llama3.2", temperature: float = 0.3,
                     max_tokens: int = 512, system: str = None) -> str:
    """Call Ollama's /api/generate endpoint."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    if system:
        payload["system"] = system

    for attempt in range(3):
        try:
            resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Ollama unreachable after 3 attempts: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")


def _groq_generate(prompt: str, model: str = "openai/gpt-oss-120b",
                   temperature: float = 0.3, max_tokens: int = 512,
                   system: str = None, reasoning_effort: str = "low") -> str:
    """Call Groq's OpenAI-compatible API."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Try loading from .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment or .env file")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # gpt-oss-* are reasoning models: reasoning tokens are billed against
    # max_tokens BEFORE any visible content is emitted. At max_tokens=512 a
    # single reply consumed 510 reasoning tokens and returned an empty string
    # with finish_reason="length" — a silent generation failure that showed up
    # downstream as a blank reply. Keep reasoning short for generation calls.
    if model.startswith("openai/gpt-oss") and reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    attempts = 6
    for attempt in range(attempts):
        try:
            resp = requests.post(f"{GROQ_BASE}/chat/completions",
                                 json=payload, headers=headers, timeout=90)
            if resp.status_code == 429:
                # Rate limited. Groq reports exactly how long to wait; guessing
                # with a capped exponential backoff instead meant a sustained
                # tokens-per-minute limit exhausted every attempt and the loop
                # fell out of the bottom returning None, which surfaced far away
                # as "'NoneType' object has no attribute 'strip'".
                retry_after = resp.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else 2 ** (attempt + 2)
                except ValueError:
                    wait = 2 ** (attempt + 2)
                wait = min(90.0, max(1.0, wait)) + 0.5
                # Print WHICH limit was hit. Groq distinguishes tokens-per-minute
                # from tokens-per-day, and only the response body says which —
                # without it a daily-budget exhaustion is indistinguishable from
                # ordinary throttling, and you wait out a limit that will not lift.
                try:
                    why = resp.json().get("error", {}).get("message", "")[:170]
                except Exception:
                    why = resp.text[:170]
                print(f"\n  [groq 429 {model}] waiting {wait:.1f}s "
                      f"(attempt {attempt + 1}/{attempts}): {why}", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            content = (choice["message"].get("content") or "").strip()

            # Empty content because the budget was spent on reasoning. Retry once
            # with a much larger budget rather than returning "" to the caller.
            if (not content and choice.get("finish_reason") == "length"
                    and attempt < attempts - 1):
                payload["max_tokens"] = min(8192, payload["max_tokens"] * 4)
                continue

            if not content:
                raise RuntimeError(
                    f"Groq returned empty content (finish_reason="
                    f"{choice.get('finish_reason')}, max_tokens={payload['max_tokens']})"
                )
            return content
        except requests.exceptions.HTTPError as e:
            if attempt < attempts - 1 and resp.status_code in (429, 500, 502, 503):
                time.sleep(2 ** (attempt + 1))
            else:
                raise RuntimeError(f"Groq API error: {e}\nResponse: {resp.text}")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Groq error: {e}")

    # Every path through the retry loop must end in a value or an exception.
    # Falling out of the bottom previously returned None to the caller.
    raise RuntimeError(f"Groq gave no usable response after {attempts} attempts")


def generate(prompt: str, model: str = "llama3.2", temperature: float = 0.3,
             max_tokens: int = 512, system: str = None, use_groq: bool = False,
             reasoning_effort: str = "low", groq_model: str = None) -> str:
    """
    Unified generation interface.
    - use_groq=False → local Ollama
    - use_groq=True  → Groq cloud API

    `groq_model` is explicit so the LLM-as-judge can run on a DIFFERENT model
    family from the one that produced the replies. Judging a model's output with
    that same model is a known source of self-preference bias, and it was the
    prior setup here (generator and judge were both gpt-oss-120b).
    """
    if use_groq:
        return _groq_generate(prompt, model=groq_model or DEFAULT_GROQ_MODEL,
                              temperature=temperature, max_tokens=max_tokens,
                              system=system, reasoning_effort=reasoning_effort)
    else:
        return _ollama_generate(prompt, model=model, temperature=temperature,
                                max_tokens=max_tokens, system=system)


def generate_json(prompt: str, model: str = "llama3.2", temperature: float = 0.1,
                  max_tokens: int = 512, system: str = None,
                  use_groq: bool = False, reasoning_effort: str = "low",
                  groq_model: str = None) -> dict:
    """
    Generate and parse a JSON response. Handles common LLM JSON formatting issues.
    """
    if system:
        system += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation."
    else:
        system = "Respond ONLY with valid JSON. No markdown, no explanation."

    raw = generate(prompt, model=model, temperature=temperature,
                   max_tokens=max_tokens, system=system, use_groq=use_groq,
                   reasoning_effort=reasoning_effort, groq_model=groq_model)

    cleaned = raw.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {"raw_response": raw, "parse_error": True}


def check_ollama() -> bool:
    """Check if Ollama is running and has a model available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return len(models) > 0
    except Exception:
        return False


def check_groq() -> bool:
    """Check if Groq API key is available."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY="):
                        return True
    return bool(api_key)


if __name__ == "__main__":
    print("Checking Ollama...", check_ollama())
    print("Checking Groq...", check_groq())
    if check_ollama():
        resp = generate("Say hello in exactly 5 words.", temperature=0.5)
        print(f"Ollama test: {resp}")
    if check_groq():
        resp = generate("Say hello in exactly 5 words.", use_groq=True, temperature=0.5)
        print(f"Groq test: {resp}")
