"""Web search: turning proven findings into things that are worth money.

A result that is real is not automatically valuable. The search agents go and
look at what the outside world is doing with the same idea — who is paying for
it, what is already published, what is unsolved — and attach that evidence to the
finding as `Lead` rows, which `arp.impact` then scores.

Providers, in order of preference: Anthropic's server-side web search (if an
Anthropic key is present), Brave, Tavily. With no keys at all, the null provider
returns nothing and the rest of the platform carries on — impact scores just fall
back to the evidence the platform gathered itself.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from arp.config import BRAVE_API_KEY, TAVILY_API_KEY
from arp.evolution import tokenize
from arp.models import Finding, Hypothesis, Lead, Subject
from arp.netutil import HttpError, get_json, post_json
from arp.store import Store

# Domains whose claims are worth more per unit of text. Deliberately short and
# editable: the point is that authority is an explicit, inspectable input.
AUTHORITY = {
    "arxiv.org": 0.9, "openreview.net": 0.9, "nature.com": 0.95, "science.org": 0.95,
    "acm.org": 0.85, "ieee.org": 0.85, "nips.cc": 0.9, "mlr.press": 0.85,
    "github.com": 0.7, "huggingface.co": 0.7, "pytorch.org": 0.75,
    "anthropic.com": 0.8, "openai.com": 0.75, "deepmind.com": 0.8, "research.google": 0.8,
    "nvidia.com": 0.75, "aws.amazon.com": 0.7, "cloud.google.com": 0.7,
    "gartner.com": 0.7, "mckinsey.com": 0.65, "a16z.com": 0.6,
    "news.ycombinator.com": 0.45, "reddit.com": 0.3, "medium.com": 0.35, "substack.com": 0.4,
}

VALUE_WORDS = (
    "cost", "costs", "pricing", "price", "revenue", "savings", "save", "budget", "roi",
    "throughput", "latency", "efficiency", "customers", "adoption", "market", "demand",
    "benchmark", "production", "deploy", "sla", "spend",
)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    source: str = "web"
    published: str = ""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class SearchProvider:
    name = "null"
    available = False

    def search(self, query: str, max_results: int = 5) -> List[SearchHit]:
        return []


class BraveProvider(SearchProvider):
    name = "brave"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or BRAVE_API_KEY

    @property
    def available(self) -> bool:  # type: ignore[override]
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> List[SearchHit]:
        try:
            data = get_json(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            )
        except HttpError:
            return []
        results = (data.get("web") or {}).get("results") or []
        return [
            SearchHit(
                title=r.get("title", ""), url=r.get("url", ""),
                snippet=re.sub(r"<[^>]+>", "", r.get("description", "") or ""),
                source="brave", published=r.get("age", "") or "",
            )
            for r in results[:max_results]
            if r.get("url")
        ]


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or TAVILY_API_KEY

    @property
    def available(self) -> bool:  # type: ignore[override]
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> List[SearchHit]:
        try:
            data = post_json(
                "https://api.tavily.com/search",
                {"api_key": self.api_key, "query": query, "max_results": max_results,
                 "search_depth": "advanced"},
            )
        except HttpError:
            return []
        return [
            SearchHit(
                title=r.get("title", ""), url=r.get("url", ""),
                snippet=r.get("content", "") or "", source="tavily",
                published=r.get("published_date", "") or "",
            )
            for r in (data.get("results") or [])[:max_results]
            if r.get("url")
        ]


class AnthropicWebProvider(SearchProvider):
    """Uses Claude's server-side web_search tool and keeps the citations."""

    name = "anthropic"

    def __init__(self, api_key: str = "", model: str = ""):
        from arp.agents import LLMClient

        self.client = LLMClient(api_key=api_key, model=model)

    @property
    def available(self) -> bool:  # type: ignore[override]
        return bool(self.client.api_key)

    def search(self, query: str, max_results: int = 5) -> List[SearchHit]:
        if not self.available:
            return []
        payload = {
            "model": self.client.model,
            "max_tokens": 1500,
            "messages": [{
                "role": "user",
                "content": (
                    f"Search the web for: {query}\n\n"
                    "Summarise what you find in two sentences. Prefer primary sources."
                ),
            }],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        }
        try:
            data = post_json(
                f"{self.client.base_url}/v1/messages",
                payload,
                headers={"x-api-key": self.client.api_key, "anthropic-version": "2023-06-01"},
            )
        except HttpError:
            return []
        return _hits_from_anthropic(data, max_results)


def _hits_from_anthropic(data: Dict[str, Any], max_results: int) -> List[SearchHit]:
    """Flatten web_search_tool_result blocks and inline citations into hits."""
    hits: List[SearchHit] = []
    seen = set()

    def add(url: str, title: str, snippet: str) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        hits.append(SearchHit(title=title or url, url=url, snippet=snippet, source="anthropic"))

    for block in data.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "web_search_tool_result":
            for item in block.get("content", []) or []:
                if isinstance(item, dict):
                    add(item.get("url", ""), item.get("title", ""), item.get("page_age", "") or "")
        for citation in block.get("citations", []) or []:
            if isinstance(citation, dict):
                add(citation.get("url", ""), citation.get("title", ""), citation.get("cited_text", "") or "")
    return hits[:max_results]


def get_provider(name: Optional[str] = None) -> SearchProvider:
    """Pick a provider: explicit name, else the first configured one, else null."""
    providers = {
        "anthropic": AnthropicWebProvider,
        "brave": BraveProvider,
        "tavily": TavilyProvider,
        "null": SearchProvider,
    }
    if name:
        provider = providers.get(name, SearchProvider)()
        return provider
    for key, factory in (("anthropic", AnthropicWebProvider), ("brave", BraveProvider), ("tavily", TavilyProvider)):
        candidate = factory()
        if candidate.available:
            return candidate
    return SearchProvider()


# ---------------------------------------------------------------------------
# Query construction and scoring
# ---------------------------------------------------------------------------

def build_queries(
    subject: Subject, finding: Optional[Finding] = None, hypothesis: Optional[Hypothesis] = None,
    limit: int = 4,
) -> List[str]:
    """Deterministic queries: what is this, who else did it, what is it worth."""
    topic = subject.title.strip()
    queries = [
        f"{topic} state of the art {datetime.now(timezone.utc).year}",
        f"{topic} commercial value cost savings",
    ]
    if hypothesis is not None:
        change = hypothesis.title.split("(")[0].strip()
        queries.insert(0, f"{change} {subject.metric} results")
        queries.append(f"{change} production deployment tradeoffs")
    if finding is not None and finding.effect:
        queries.append(f"{topic} {abs(finding.effect):.1%} improvement {subject.metric} benchmark")
    seen, out = set(), []
    for q in queries:
        q = " ".join(q.split())
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:limit]


def domain_of(url: str) -> str:
    match = re.match(r"https?://([^/]+)", url or "")
    host = match.group(1).lower() if match else ""
    return host[4:] if host.startswith("www.") else host


def authority_of(url: str) -> float:
    host = domain_of(url)
    for domain, score in AUTHORITY.items():
        if host == domain or host.endswith("." + domain):
            return score
    if host.endswith(".edu") or host.endswith(".ac.uk"):
        return 0.8
    if host.endswith(".gov"):
        return 0.85
    return 0.4


def score_hit(subject: Subject, hit: SearchHit, query: str) -> Dict[str, float]:
    """Relevance / authority / commercial signal, all in [0, 1] and all explainable."""
    text = f"{hit.title} {hit.snippet}".lower()
    subject_terms = set(tokenize(subject.title) + tokenize(subject.description) + tokenize(query))
    hit_terms = set(tokenize(text))
    overlap = len(subject_terms & hit_terms) / max(len(subject_terms), 1)
    commercial = sum(1 for w in VALUE_WORDS if w in text) / len(VALUE_WORDS)
    year_match = re.search(r"\b(20\d{2})\b", f"{hit.published} {text}")
    recency = 0.5
    if year_match:
        age = datetime.now(timezone.utc).year - int(year_match.group(1))
        recency = max(0.0, min(1.0, 1.0 - age / 6.0))
    return {
        "relevance": round(min(1.0, overlap * 2.0), 4),
        "authority": round(authority_of(hit.url), 4),
        "commercial": round(min(1.0, commercial * 4.0), 4),
        "recency": round(recency, 4),
    }


def gather_leads(
    store: Store,
    subject: Subject,
    finding: Optional[Finding] = None,
    hypothesis: Optional[Hypothesis] = None,
    provider: Optional[SearchProvider] = None,
    max_per_query: int = 5,
) -> List[Lead]:
    """Run the queries, score the hits, persist them as leads."""
    provider = provider or get_provider()
    if not provider.available:
        return []
    leads: List[Lead] = []
    for query in build_queries(subject, finding, hypothesis):
        for hit in provider.search(query, max_per_query):
            lead = Lead(
                subject_id=subject.id,
                finding_id=finding.id if finding else None,
                title=hit.title[:300],
                url=hit.url,
                snippet=(hit.snippet or "")[:1000],
                source=hit.source,
                query=query,
                scores=score_hit(subject, hit, query),
            )
            leads.append(store.add_lead(lead))
    return leads


def market_signal(leads: Sequence[Lead]) -> float:
    """Aggregate demand evidence from leads into a single [0, 1] number.

    Weighted by authority, so a vendor benchmark page counts for less than a
    peer-reviewed result saying the same thing.
    """
    if not leads:
        return 0.0
    num = den = 0.0
    for lead in leads:
        s = lead.scores or {}
        weight = float(s.get("authority", 0.4)) * (0.5 + 0.5 * float(s.get("recency", 0.5)))
        signal = 0.6 * float(s.get("commercial", 0.0)) + 0.4 * float(s.get("relevance", 0.0))
        num += weight * signal
        den += weight
    return round(min(1.0, num / den if den else 0.0), 4)
