"""hls.prompts -- prompt template assets for the HLS harness.

The default implementer brief lives at ``brief.jinja`` in this package and is
loaded at import time via ``importlib.resources``.  Task templates that
``{% extends "brief" %}`` continue to work because ``prompt.py`` registers the
content in the shared Jinja ``DictLoader`` under the key ``"brief"``.
"""
