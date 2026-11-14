import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from urllib.parse import urljoin, urlparse, quote_plus

app = FastAPI(title="DeepSearch OSINT Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    keywords: Optional[List[str]] = Field(default_factory=list)
    max_depth: int = Field(default=1, ge=1, le=3)
    max_pages: int = Field(default=5, ge=1, le=25)
    gemini_api_key: Optional[str] = None

class Artifact(BaseModel):
    title: str
    source_url: str
    snippet: str

class SearchResponse(BaseModel):
    visited_count: int
    artifacts_count: int
    artifacts: List[Artifact]
    abductive_ai_synthesis: str

async def fetch_duckduckgo_seeds(session: aiohttp.ClientSession, query: str, max_results: int) -> List[str]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    urls = []
    try:
        async with session.get(url, headers=headers, timeout=12) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.select("a.result__url"):
                    href = a.get("href", "")
                    if href.startswith("http"):
                        urls.append(href)
                    if len(urls) >= max_results:
                        break
    except Exception:
        pass
    return urls

async def scrape_page(session: aiohttp.ClientSession, url: str) -> Optional[Artifact]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DeepSearchOSINT/1.0)"}
    try:
        async with session.get(url, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                return None
            text = await resp.text(errors="ignore")
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            body_text = " ".join(soup.stripped_strings)
            snippet = body_text[:400] if body_text else "No readable text extracted."
            return Artifact(title=title[:120], source_url=url, snippet=snippet)
    except Exception:
        return None

async def synthesize_with_gemini(session: aiohttp.ClientSession, query: str, artifacts: List[Artifact], api_key: str) -> str:
    if not api_key:
        return "No Gemini API key provided. Raw artifacts returned below."
    if not artifacts:
        return "No web artifacts were recovered for this query."

    context_lines = [f"[{i+1}] {a.title} ({a.source_url}): {a.snippet}" for i, a in enumerate(artifacts)]
    prompt = (
        f"Perform an abductive OSINT analysis on the following scraped web artifacts for query: '{query}'. "
        f"Identify key patterns, anomalies, and actionable findings concisely:\n\n" + "\n".join(context_lines)
    )

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with session.post(gemini_url, json=payload, timeout=20) as resp:
            data = await resp.json()
            if resp.status == 200 and "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return f"Synthesis API Notice ({resp.status}): {data.get('error', {}).get('message', 'Check API Key format.')}"
    except Exception as e:
        return f"Synthesis connection error: {str(e)}"

@app.get("/health")
async def health():
    return {"status": "online"}

@app.post("/api/deep-search", response_model=SearchResponse)
async def deep_search(req: SearchRequest):
    api_key = (req.gemini_api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        seeds = await fetch_duckduckgo_seeds(session, req.query, req.max_pages)
        tasks = [scrape_page(session, u) for u in seeds[:req.max_pages]]
        scraped = await asyncio.gather(*tasks)
        artifacts = [a for a in scraped if a is not None]
        synthesis = await synthesize_with_gemini(session, req.query, artifacts, api_key)
        return SearchResponse(
            visited_count=len(seeds),
            artifacts_count=len(artifacts),
            artifacts=artifacts,
            abductive_ai_synthesis=synthesis
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
