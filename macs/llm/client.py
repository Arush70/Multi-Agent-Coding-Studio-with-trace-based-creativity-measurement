"""OpenAI-compatible chat client with content-addressed caching.

Arush - this is the single door every model call in your project goes through.
Two things make it worth having instead of calling the API directly:

  1. Caching. Every call is saved to disk under a hash of exactly what was asked
     (model, revision, messages, temperature, seed, max_tokens). Ask the same
     thing again and you get the saved answer for free - no GPU, no waiting. This
     is what lets you re-run your whole analysis a hundred times while writing the
     report without paying for a single extra generation, and it is what makes
     "reproducible" true rather than aspirational.

  2. One interface for every backend. Ollama now, a Kaggle GPU later, DeepSeek's
     API for the frontier check - all speak this same protocol. Swapping between
     them is a change to config/models.yaml, never a change to your code. That is
     why nothing above this file ever imports 'requests' or knows a URL.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

# Matches ${NAME} in the config so an API key can live in an environment variable
# instead of in the file. This is the mechanism that keeps your DeepSeek key off
# GitHub: models.yaml says ${DEEPSEEK_API_KEY}, never the key itself.
_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    return value


@dataclass
class ModelProfile:
    name: str
    base_url: str
    api_key: str
    model: str
    revision: str = ""
    quantisation: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048

    @property
    def identity(self) -> dict[str, Any]:
        # This dict is stamped into the header of every session trace. When a marker
        # or your future self asks "which model produced this result?", the answer is
        # here - including the revision hash you pinned (dae161e27b0e for your local
        # Qwen). Without this, your traces would be un-reproducible and that costs you
        # marks under the Rigour criterion.
        return {
            "profile": self.name,
            "model": self.model,
            "revision": self.revision,
            "quantisation": self.quantisation,
            "temperature": self.temperature,
        }


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    request_hash: str
    cached: bool          # True = served from disk, no backend call happened
    latency_s: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMClient:
    def __init__(self, profile: ModelProfile, cache_dir: str | Path = "data/cache", seed: int = 0):
        self.profile = profile
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    @classmethod
    def from_config(cls, path: str | Path = "config/models.yaml", profile: str | None = None) -> "LLMClient":
        # Reads config/models.yaml (the git-ignored one, not the .example). Pass
        # profile="local" / "kaggle" / "deepseek" to pick a backend; omit it and the
        # file's default_profile is used. This is the only place a profile is built.
        cfg = yaml.safe_load(Path(path).read_text())
        name = profile or cfg["default_profile"]
        raw = {k: _expand_env(v) for k, v in cfg["profiles"][name].items()}
        prof = ModelProfile(name=name, **raw)
        return cls(prof, cache_dir=cfg.get("cache_dir", "data/cache"), seed=int(cfg.get("seed", 0)))

    def request_hash(self, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int) -> str:
        # The cache key. Note revision is inside it: if you ever change the model
        # version, every hash changes and nothing stale is silently reused. That is
        # deliberate - a cache that ignored the model version would quietly mix
        # results from two different models and ruin an experiment.
        payload = {
            "model": self.profile.model,
            "revision": self.profile.revision,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, h: str) -> Path:
        # Files are bucketed by the first two hex chars (data/cache/ab/abcd...json)
        # so one folder never fills with 70,000 files - some tools choke on that.
        return self.cache_dir / h[:2] / f"{h}.json"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        use_cache: bool = True,
        retries: int = 3,
    ) -> Completion:
        temperature = self.profile.temperature if temperature is None else temperature
        max_tokens = self.profile.max_tokens if max_tokens is None else max_tokens
        seed = self.seed if seed is None else seed
        h = self.request_hash(messages, temperature, max_tokens, seed)
        path = self._cache_path(h)

        # Cache hit: return immediately, cached=True, no network. This branch is what
        # your reruns spend 99% of their time in once an experiment has been run once.
        if use_cache and path.exists():
            data = json.loads(path.read_text())
            return Completion(
                text=data["text"],
                prompt_tokens=data["prompt_tokens"],
                completion_tokens=data["completion_tokens"],
                request_hash=h,
                cached=True,
                latency_s=0.0,
                raw=data.get("raw", {}),
            )

        body = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        headers = {"Authorization": f"Bearer {self.profile.api_key}", "Content-Type": "application/json"}
        url = self.profile.base_url.rstrip("/") + "/chat/completions"

        # Retries with backoff (1s, 2s, 4s). A local Ollama rarely needs this, but a
        # Kaggle server under load or a flaky network will drop the occasional call,
        # and you do not want a 12-hour reference run to die on one hiccup.
        last_err: Exception | None = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = self._session.post(url, json=body, headers=headers, timeout=300)
                r.raise_for_status()
                raw = r.json()
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

        text = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})
        comp = Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            request_hash=h,
            cached=False,
            latency_s=time.time() - t0,
            raw=raw,
        )
        # Write the cache entry. We store the full raw response and the exact request
        # too, not just the text - if a reviewer ever questions a number, you can show
        # precisely what the model was asked and what it returned, byte for byte.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "text": comp.text,
            "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens,
            "raw": raw,
            "request": body,
            "profile": self.profile.identity,
        }, ensure_ascii=False, indent=1))
        return comp


def prompt_hash(text: str) -> str:
    # Every role prompt (Innovator, Critic, ...) gets hashed and the hash goes in the
    # session header. If you tweak a prompt mid-project, the hash changes and your
    # traces show exactly which sessions used which wording. This is how you avoid the
    # classic disaster of "I changed the prompt halfway and can no longer tell which
    # results are which".
    return hashlib.sha256(text.encode()).hexdigest()[:16]
