"""Smoke test for the model client and the response cache.

    python scripts/smoke_llm.py

Sends one prompt to the configured model twice. The first call is a real generation;
the second is served from data/cache without a model call ('cached=True'). Requires a
running model server, so it is a manual script rather than a pytest test.
"""
from macs.llm.client import LLMClient


def main() -> None:
    # profile="local" -> the Ollama profile in config/models.yaml.
    client = LLMClient.from_config(profile="local")
    messages = [
        {"role": "system", "content": "You are a terse Python assistant."},
        {"role": "user", "content": "Write a one-line function is_even(n)."},
    ]

    first = client.chat(messages, seed=1)
    print(f"[1] cached={first.cached}  latency={first.latency_s:.2f}s  "
          f"tokens={first.completion_tokens}  hash={first.request_hash[:12]}")
    print("---- model output ----")
    print(first.text.strip())
    print("----------------------")

    # Same request, same seed -> must come from cache.
    second = client.chat(messages, seed=1)
    print(f"[2] cached={second.cached}  latency={second.latency_s:.2f}s   "
          f"(same request, so this should say cached=True)")

    # Change the seed -> new request -> real call again, proving the seed is part of
    # the identity of a generation (repetitions differ only by seed).
    third = client.chat(messages, seed=2)
    print(f"[3] cached={third.cached}  (different seed, so a fresh generation)")


if __name__ == "__main__":
    main()
