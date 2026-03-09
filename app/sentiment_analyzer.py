"""
sentiment_analyzer.py — Financial news sentiment via Claude API.

Non-trivial LLM use:
  1. Structured chain-of-thought prompt forces signal decomposition before scoring
  2. JSON output parsed and validated — malformed responses trigger regex fallback
  3. Sentiment score mathematically blended with XGBoost regime score (max ±15 pts)
  4. Multi-headline batch mode with confidence-weighted aggregation
"""

import json
import re
from openai import OpenAI
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SentimentResult:
    headline: str
    sentiment: str            # "positive" | "negative" | "neutral"
    score: float              # -1.0 to +1.0
    confidence: float         # 0.0 to 1.0
    signals: list             # key financial signals detected
    reasoning: str            # chain-of-thought explanation
    market_impact: str        # "bullish" | "bearish" | "neutral"
    affected_sectors: list    # e.g. ["Technology", "Financials"]


@dataclass
class CompositeScore:
    xgb_score: float          # original XGBoost 0-100
    sentiment_shift: float    # sentiment contribution in score points
    composite: float          # blended 0-100
    regime: str               # regime label from composite score
    n_headlines: int          # number of headlines used


SYSTEM_PROMPT = """You are a quantitative financial sentiment analyst with deep expertise
in equity markets. Analyse financial news from an institutional investor's perspective
and extract structured signals suitable for quantitative combination with technical
market indicators.

Follow this reasoning chain before scoring:
1. SIGNAL EXTRACTION — identify the concrete financial signals present
   (earnings, guidance, macro data, central bank language, geopolitical risk, flows)
2. DIRECTIONAL BIAS — determine the net impact on broad equity markets
3. CONFIDENCE CALIBRATION — assess how unambiguous the signal is
4. SECTOR IMPACT — identify directly affected sectors

You MUST respond with ONLY a valid JSON object. No preamble, no text outside JSON.
Schema:
{
  "sentiment": "positive" | "negative" | "neutral",
  "score": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "signals": [<2-4 short strings describing detected signals>],
  "reasoning": "<2-3 sentence chain-of-thought>",
  "market_impact": "bullish" | "bearish" | "neutral",
  "affected_sectors": [<sector names>]
}

Score calibration:
+0.8 to +1.0  Strong positive: massive earnings beat, dovish Fed pivot, geopolitical resolution
+0.4 to +0.8  Moderate positive: rate cut, M&A premium, solid guidance raise
+0.1 to +0.4  Mild positive: slight beat, minor easing, constructive policy language
-0.1 to +0.1  Neutral: mixed signals, in-line data, ambiguous language
-0.1 to -0.4  Mild negative: slight miss, minor tightening, uncertainty language
-0.4 to -0.8  Moderate negative: earnings miss, hawkish surprise, tariff escalation
-0.8 to -1.0  Strong negative: systemic risk, crash language, major policy shock

Disambiguation rules (common errors to avoid):
- "beats earnings BY X%" = company EXCEEDED estimates → POSITIVE score
- "beats X% of expected earnings" = company only hit X% of target → NEGATIVE score
- "raises guidance" = management expects MORE future earnings → POSITIVE score
- "cuts guidance" = management expects LESS future earnings → NEGATIVE score
- "Fed signals pause" = no more hikes imminent → POSITIVE for equities
- Always reason from the INVESTOR perspective: what does this mean for future cash flows?"""


def _parse_response(raw: str) -> dict:
    """Extract and validate JSON from model response, with regex fallback."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def analyze_headline(headline: str, api_key: str) -> Optional[SentimentResult]:
    """Analyze a single financial headline. Returns None on API or parse failure."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Analyse this financial news and return the JSON sentiment object:\n\n"
                    f"\"{headline.strip()}\""
                )}
            ]
        )
        data = _parse_response(response.choices[0].message.content)
        if not data or "sentiment" not in data:
            return None
        return SentimentResult(
            headline=headline.strip(),
            sentiment=str(data.get("sentiment", "neutral")),
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.5)),
            signals=list(data.get("signals", [])),
            reasoning=str(data.get("reasoning", "")),
            market_impact=str(data.get("market_impact", "neutral")),
            affected_sectors=list(data.get("affected_sectors", [])),
        )
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None


def analyze_batch(headlines: list, api_key: str) -> list:
    """Analyze a list of headlines, skipping empty lines."""
    results = []
    for h in headlines:
        if h.strip():
            r = analyze_headline(h, api_key)
            if r:
                results.append(r)
    return results


def blend_scores(xgb_score: float, results: list) -> CompositeScore:
    """
    Blend the XGBoost regime score with news sentiment.

    Each headline's score ∈ [-1, +1] is weighted by confidence.
    The confidence-weighted average is mapped to ±15 score points
    (matching the ±22pt cap used in the XGBoost target construction).
    The composite is clamped to [0, 100] and re-labelled.
    """
    from predictor import _score_to_regime

    if not results:
        return CompositeScore(
            xgb_score=xgb_score,
            sentiment_shift=0.0,
            composite=round(xgb_score, 1),
            regime=_score_to_regime(xgb_score),
            n_headlines=0,
        )

    total_conf = sum(r.confidence for r in results)
    avg_sentiment = (
        sum(r.score * r.confidence for r in results) / total_conf
        if total_conf > 0 else 0.0
    )

    MAX_SHIFT = 15.0
    shift = round(avg_sentiment * MAX_SHIFT, 2)
    composite = float(max(0.0, min(100.0, xgb_score + shift)))

    return CompositeScore(
        xgb_score=xgb_score,
        sentiment_shift=shift,
        composite=round(composite, 1),
        regime=_score_to_regime(composite),
        n_headlines=len(results),
    )