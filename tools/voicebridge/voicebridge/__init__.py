"""voicebridge — speak into Meta glasses, get answers from Claude, ChatGPT, Perplexity or Cursor."""

from .bridge import Bridge, Exchange
from .config import TARGETS, Config
from .router import Reply, Router, parse

__version__ = "0.1.0"
__all__ = ["Bridge", "Exchange", "Config", "TARGETS", "Reply", "Router", "parse", "__version__"]
