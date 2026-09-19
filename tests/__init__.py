"""Tests for the agentic research platform (`arp`).

Run them with:

    python3 -m unittest discover -s tests -t .

No GPU, no network and no third-party packages required: every test drives the
platform through the seeded `SimulatedRunner`, whose response surface has a known
ground truth, so the suite can assert that real effects get proven and null
effects never do.
"""
