"""Supervised fine-tuning pipeline for the library agent.

``cli`` is ``library-train`` (prepare, sweep, select, run). ``prepare``
normalizes public tool-calling JSONL. ``train`` runs PEFT LoRA via TRL.
``sweep`` drives W&B trials. ``select`` picks a candidate under cost and
quality constraints. ``merge`` and ``publish`` support the deploy CLI.
"""
