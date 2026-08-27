"""
Pro-Router serving package: scheduler, model actors, verifier gate, pre-scorer.

Drives speculative decoding across two vLLM instances (draft + target) via
Ray actors. See prorouter/README.md for usage.
"""
