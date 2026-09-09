"""
Thin wrapper around the Ollama HTTP API and optional Groq API for LLM-as-judge.
Supports both local Ollama and cloud Groq inference.
"""

import json
import re
import time
import requests
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


OLLAMA_BASE = "http://localhost:11434"
GROQ_BASE = "https://api.groq.com/openai/v1"

# Which Groq model the agent uses. Overridable via GROQ_MODEL so a run can move
# to another model without editing code — necessary in practice because each
# model carries its own per-day token budget on the free tier, and exhausting one
# should not block the benchmark. The committed benchmark artifacts were produced
# with this default; results/run_config_*.json records what actually ran.
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


class _TokenPacer:
    """
    Client-side pacing against Groq's tokens-per-minute budget.

    Groq returns the remaining TPM budget and its refill time on every response.
    Without using them the only way to discover the limit is to hit a 429 and
    back off, and because a single request can need more tokens than the bucket
    refills in the server's suggested wait, that path burns far more wall time
    than the limit itself requires. Reserving against the last known headers
    turns a retry storm into a short, precise sleep.
    """

    def __init__(self):
        self.remaining: dict[str, float] = {}
        self.reset_at: dict[str, float] = {}

    @staticmethod
    def _parse_duration(text: str) -> float:
        """Parse Groq's '1m2.5s' / '56.19s' / '615ms' reset format into seconds."""
        if not text:
            return 0.0
        total, num = 0.0, ""
        i = 0
        while i < len(text):
            c = text[i]
            if c.isdigit() or c == ".":
                num += c
                i += 1
            elif text[i:i + 2] == "ms":
                total += float(num or 0) / 1000.0
                num = ""; i += 2
            elif c == "h":
                total += float(num or 0) * 3600; num = ""; i += 1
            elif c == "m":
                total += float(num or 0) * 60; num = ""; i += 1
            elif c == "s":
                total += float(num or 0); num = ""; i += 1
            else:
                i += 1
        return total

    def wait_if_needed(self, model: str, estimated_tokens: float) -> None:
        remaining = self.remaining.get(model)
        if remaining is None or remaining >= estimated_tokens:
            return
        delay = max(0.0, self.reset_at.get(model, 0.0) - time.time())
        if delay <= 0:
            return
        print(f"  [pacing {model}] {remaining:.0f} tokens left, need ~{estimated_tokens:.0f}; "
              f"sleeping {delay:.1f}s", flush=True)
        time.sleep(delay + 0.3)
        # Assume a full bucket after the reset window elapses.
        self.remaining[model] = None

    def observe(self, model: str, headers) -> None:
        rem = headers.get("x-ratelimit-remaining-tokens")
        rst = headers.get("x-ratelimit-reset-tokens")
        if rem is not None:
            try:
                self.remaining[model] = float(rem)
            except ValueError:
                self.remaining[model] = None
        if rst:
            self.reset_at[model] = time.time() + self._parse_duration(rst)


_PACER = _TokenPacer()


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


from src.rate_limiter import default_rate_limiter, GroqRateLimitExceeded


def _groq_generate(prompt: str, model: str = "openai/gpt-oss-120b",
                   temperature: float = 0.3, max_tokens: int = 512,
                   system: str = None, reasoning_effort: str = "low") -> str:
    """Call Groq's OpenAI-compatible API."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
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
    if model.startswith("openai/gpt-oss") and reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    prompt_chars = sum(len(m["content"]) for m in messages)
    estimated = int(prompt_chars / 4.0 + max_tokens)

    attempts = default_rate_limiter.max_retries
    for attempt in range(attempts):
        try:
            default_rate_limiter.acquire(estimated_tokens=estimated)
            resp = requests.post(f"{GROQ_BASE}/chat/completions",
                                 json=payload, headers=headers, timeout=90)
            default_rate_limiter.observe_response(resp.headers)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                try:
                    why = resp.json().get("error", {}).get("message", "")
                except Exception:
                    why = resp.text

                if "per day" in why.lower() or "(tpd)" in why.lower():
                    raise RuntimeError(
                        f"Groq daily token budget exhausted for {model}.\n"
                        f"  {why[:200]}\n"
                        f"  This resets on a 24h cycle. Either wait, set "
                        f"GROQ_MODEL to a model with remaining budget, or run "
                        f"with --local against Ollama."
                    )

                if attempt < attempts - 1:
                    print(f"\n  Groq rate limit reached — backing off (attempt {attempt + 1}/{attempts})", flush=True)
                    wait = default_rate_limiter.handle_429(attempt, retry_after=retry_after)
                    time.sleep(wait)
                    continue
                else:
                    print(f"\n  Groq unavailable after {attempts} attempts; example marked failed.", flush=True)
                    raise GroqRateLimitExceeded(
                        f"Groq rate limit exceeded after {attempts} attempts for {model}: {why[:150]}"
                    )

            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            content = (choice["message"].get("content") or "").strip()

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
                time.sleep(2.0 ** (attempt + 1))
            else:
                raise RuntimeError(f"Groq API error: {e}\nResponse: {resp.text}")
        except (RuntimeError, GroqRateLimitExceeded):
            raise
        except Exception as e:
            if attempt < attempts - 1:
                time.sleep(2.0 ** attempt)
            else:
                raise RuntimeError(f"Groq error: {e}")

    raise GroqRateLimitExceeded(f"Groq gave no usable response after {attempts} attempts")


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

    try:
        raw = generate(prompt, model=model, temperature=temperature,
                       max_tokens=max_tokens, system=system, use_groq=use_groq,
                       reasoning_effort=reasoning_effort, groq_model=groq_model)
    except (GroqRateLimitExceeded, RuntimeError) as e:
        return {"raw_response": "", "error": str(e), "parse_error": True, "failed": True}

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
