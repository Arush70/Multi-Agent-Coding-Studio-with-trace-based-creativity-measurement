"""OpenAI-compatible chat client with content-addressed caching.

Every model call goes through this client, for two reasons:

  1. Caching. Every call is saved to disk under a hash of exactly what was asked (model,
     revision, messages, temperature, seed, max_tokens). Repeating a request returns the
     saved answer without a model call, so analyses can be re-run at no cost and runs are
     reproducible.

  2. One interface for every backend. A local Ollama server, a hosted GPU, or an API such as
     DeepSeek's all speak the same protocol; switching between them is a change to
     config/models.yaml, not to the code. Nothing above this module imports 'requests' or
     knows a URL.
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

# Matches ${NAME} in the config so an API key can live in an environment variable instead
# of in the file: models.yaml says ${DEEPSEEK_API_KEY}, never the key itself.
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
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> dict[str, Any]:
        # Stamped into the header of every session trace, so each result records exactly
        # which model, pinned revision, quantisation and temperature produced it.
        ident = {
            "profile": self.name,
            "model": self.model,
            "revision": self.revision,
            "quantisation": self.quantisation,
            "temperature": self.temperature,
        }
        if self.extra_body:
            ident["extra_body"] = self.extra_body
        return ident


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
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        name = profile or cfg["default_profile"]
        raw = {k: _expand_env(v) for k, v in cfg["profiles"][name].items()}
        prof = ModelProfile(name=name, **raw)
        return cls(prof, cache_dir=cfg.get("cache_dir", "data/cache"), seed=int(cfg.get("seed", 0)))

    def request_hash(self, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int) -> str:
        # The cache key. The revision is part of it, so changing the model version changes
        # every hash and no stale response is reused; a cache that ignored the version would
        # silently mix results from two different models.
        payload = {
            "model": self.profile.model,
            "revision": self.profile.revision,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        # Backend options (for example switching off a model's thinking mode) change the
        # output, so they are part of the key. They are added only when set, so caches built
        # without them keep their hashes.
        if self.profile.extra_body:
            payload["extra_body"] = self.profile.extra_body
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

        # Cache hit: return immediately, cached=True, no network call. An unreadable entry
        # (e.g. left half-written by a crash) counts as a miss and is rewritten below.
        data = _read_cache(path) if use_cache else None
        if data is not None:
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
            **self.profile.extra_body,
        }
        headers = {"Authorization": f"Bearer {self.profile.api_key}", "Content-Type": "application/json"}
        url = self.profile.base_url.rstrip("/") + "/chat/completions"

        # Retries with backoff (1s, 2s, 4s). A local server rarely needs this, but a remote
        # server under load or a flaky network can drop the occasional call, and a long
        # reference run should not fail on one transient error.
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

        text = _strip_thinking(raw["choices"][0]["message"].get("content") or "")
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
        # Write the cache entry. The full raw response and the exact request are stored,
        # not just the text, so any result can be traced to precisely what the model was
        # asked and what it returned.
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, json.dumps({
            "text": comp.text,
            "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens,
            "raw": raw,
            "request": body,
            "profile": self.profile.identity,
        }, ensure_ascii=False, indent=1))
        return comp


def _read_cache(path: Path) -> dict[str, Any] | None:
    """Read a cache entry as UTF-8. Entries written before 2026-10-15 on Windows used the
    system code page, so that is tried next. Returns None when the entry is missing or
    unreadable."""
    if not path.exists():
        return None
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252"):
        try:
            data = json.loads(raw.decode(enc))
            if isinstance(data, dict) and "text" in data:
                return data
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def _write_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text to a temporary file and move it into place, so a crash never leaves a
    half-written cache entry."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove a leading <think>...</think> block that some reasoning models put in the answer.

    The full response, including any reasoning, is still stored in the cache entry.
    """
    return _THINK_RE.sub("", text, count=1)


def prompt_hash(text: str) -> str:
    # Every role prompt (Innovator, Critic, ...) is hashed and the hash goes in the session
    # header, so a prompt change is visible in the traces and each session records exactly
    # which wording it used.
    return hashlib.sha256(text.encode()).hexdigest()[:16]
