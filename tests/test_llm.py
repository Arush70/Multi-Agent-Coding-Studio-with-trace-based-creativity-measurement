"""Tests for the model client.

These run with no model server and no network, so they can run in CI. The backend's HTTP
response is faked, and the tests check the cache key, the caching behaviour, and reading the
API key from an environment variable. The end-to-end check against a live model is
scripts/smoke_llm.py.
"""
from __future__ import annotations

from macs.llm.client import LLMClient, ModelProfile, prompt_hash


def _profile() -> ModelProfile:
    return ModelProfile(
        name="test", base_url="http://localhost:9/v1", api_key="none",
        model="qwen2.5-coder:7b-instruct-q4_K_M", revision="dae161e27b0e",
    )


class _FakeResponse:
    # Stands in for what a real backend returns, so no server is needed.
    def __init__(self, text: str):
        self._text = text
        self.call_count = 0

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [{"message": {"content": self._text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }


def test_hash_is_deterministic_and_sensitive(tmp_path):
    c = LLMClient(_profile(), cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    h1 = c.request_hash(msgs, 0.7, 128, 1)
    h2 = c.request_hash(msgs, 0.7, 128, 1)
    assert h1 == h2                                   # same input -> same key
    assert h1 != c.request_hash(msgs, 0.7, 128, 2)    # different seed -> different key
    assert h1 != c.request_hash(msgs, 0.9, 128, 1)    # different temperature -> different key


def test_cache_hit_avoids_second_backend_call(tmp_path, monkeypatch):
    c = LLMClient(_profile(), cache_dir=tmp_path)
    fake = _FakeResponse("def f(): return 1")

    def fake_post(url, json, headers, timeout):
        fake.call_count += 1
        return fake

    monkeypatch.setattr(c._session, "post", fake_post)
    msgs = [{"role": "user", "content": "write f"}]

    first = c.chat(msgs, seed=1)
    assert first.cached is False and fake.call_count == 1   # first call hit the backend

    second = c.chat(msgs, seed=1)
    assert second.cached is True and fake.call_count == 1   # second call served from disk, backend untouched
    assert second.text == first.text


def test_env_var_expands_into_api_key(tmp_path, monkeypatch):
    # Proves the ${DEEPSEEK_API_KEY} mechanism works, i.e. the key is read from the
    # environment and never has to be written into models.yaml.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-123")
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "default_profile: deepseek\n"
        "cache_dir: %s\n"
        "profiles:\n"
        "  deepseek:\n"
        "    base_url: https://api.deepseek.com/v1\n"
        "    api_key: ${DEEPSEEK_API_KEY}\n"
        "    model: deepseek-v4-flash\n" % (tmp_path / "cache")
    )
    c = LLMClient.from_config(cfg)
    assert c.profile.api_key == "secret-123"


def test_prompt_hash_is_short_and_stable():
    assert prompt_hash("abc") == prompt_hash("abc")
    assert len(prompt_hash("abc")) == 16


def test_extra_body_is_sent_and_keyed_without_changing_old_hashes(tmp_path, monkeypatch):
    plain = LLMClient(_profile(), cache_dir=tmp_path)
    prof = _profile()
    prof.extra_body = {"reasoning_effort": "none"}
    opt = LLMClient(prof, cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    assert plain.request_hash(msgs, 0.7, 128, 1) != opt.request_hash(msgs, 0.7, 128, 1)
    assert "extra_body" not in plain.profile.identity
    assert opt.profile.identity["extra_body"] == {"reasoning_effort": "none"}

    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(json)
        return _FakeResponse("<think>\nplan\n</think>\n\nanswer")

    monkeypatch.setattr(opt._session, "post", fake_post)
    out = opt.chat(msgs, seed=1)
    assert sent["reasoning_effort"] == "none"
    assert out.text == "answer"


def test_cache_is_utf8_and_survives_legacy_or_broken_entries(tmp_path, monkeypatch):
    import json
    from macs.llm.client import _read_cache
    c = LLMClient(_profile(), cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "complexity?"}]
    monkeypatch.setattr(c._session, "post", lambda *a, **k: _FakeResponse("O(n * α(k)) ≤ n²"))
    out = c.chat(msgs, seed=1)
    path = c._cache_path(out.request_hash)
    assert json.loads(path.read_bytes().decode("utf-8"))["text"] == "O(n * α(k)) ≤ n²"
    assert not list(tmp_path.rglob("*.tmp"))

    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(json.dumps({"text": "café – x"}, ensure_ascii=False).encode("cp1252"))
    assert _read_cache(legacy)["text"] == "café – x"

    path.write_text("", encoding="utf-8")                  # half-written by a crash
    assert _read_cache(path) is None
    again = c.chat(msgs, seed=1)
    assert again.cached is False and again.text == "O(n * α(k)) ≤ n²"
    assert c.chat(msgs, seed=1).cached is True
