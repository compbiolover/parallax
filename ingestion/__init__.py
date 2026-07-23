"""Ingestion: pull content from both modeled diets into the datastore.

Sources (see ``config/sources.yaml``): RSS/Atom via feedparser (primary),
article body extraction via trafilatura, GDELT DOC 2.0 and Media Cloud for
discovery/corpus construction, podcast audio (faster-whisper transcription),
and YouTube (captions first, audio fallback).

Guardrails: honor robots.txt, rate limits, and each source's terms. Raw text
and audio are transient processing artifacts, never committed.
"""
