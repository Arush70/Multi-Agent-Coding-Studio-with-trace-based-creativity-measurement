"""First real model call through your own code - and proof the cache works.

Arush - run this once your Ollama is up (it is) and config/models.yaml points at it
(it does). It sends one prompt to your local Qwen twice. The first call is a real
generation; the second is served from data/cache with no model call. If you see
'cached=True' on the second line, your whole reproducibility foundation works.

    python scripts/smoke_llm.py

This is a smoke test for YOU, not for CI - it needs the live model, so it is a
script you run by hand, not a pytest test.
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
    # the identity of a generation (which matters: your repetitions differ only by seed).
    third = client.chat(messages, seed=2)
    print(f"[3] cached={third.cached}  (different seed, so a fresh generation)")


if __name__ == "__main__":
    main()
