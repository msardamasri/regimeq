# PHASE 1 — IDEATION: AI/Data Feature Prototype for Streamlit

## Your Role
You are a senior product designer and full-stack ML engineer helping me build an
outstanding Streamlit prototype for a master's-level "Prototyping with Data & AI" course
at ESADE. The goal is to score 9-10/10, which requires the prototype to
"significantly exceed expectations in an outstanding/mind-blowing way" across three
dimensions: Effectivity (50%), Originality (25%), and Documentation (25%).

## Context
- **Assignment:** Propose and prototype a new AI/data-powered feature for an existing
  or fictitious product. The feature must put data and/or AI at the CENTER of the value
  it delivers — not just "an app that displays data or predictions."
- **Portfolio approach:** Assignment 2 will be an increment of Assignment 1, so the idea
  should have room to grow and improve in a second iteration.
- **Benchmark example:** A past top project was a Google Maps extension that
  recalculated routes based on scenic beauty, using a CV model trained on Street View
  images to predict how visually appealing each street segment was. That's the caliber
  of "data product thinking" expected.

## Technical Constraints
- Must deploy on **Streamlit Community Cloud** (free tier: no GPU, limited memory,
  Python packages only)
- I have access to **OpenAI API** (GPT-4o-mini, embeddings, etc.) and/or
  **Anthropic API** (Claude Sonnet)
- Can use **pre-trained models** from HuggingFace (CPU-compatible only) or
  **lightweight scikit-learn/XGBoost models**
- Can use **public APIs and datasets** (Spotify, Reddit, government data, etc.)
- Where a real model isn't feasible for the prototype, a **realistic mock pipeline** is
  acceptable — but at least one core AI component should be real

## My Background
I'm an MSc Business Analytics student with a BSc in Computer Engineering. Comfortable
with Python, scikit-learn, PyTorch, pandas, Flask/FastAPI, Docker, and API integrations.
I've done LLM fine-tuning (QLoRA), RAG pipelines, and ML on AWS. I can handle
technical complexity — don't simplify for me.

## What I Need — Phase 1
Propose **exactly 5 product ideas**, each following this structure:

### For each idea:
1. **Product & Feature** — What existing product does this live in? What's the new
   AI-powered feature?
2. **User Problem** — What real pain point does this solve? Why would someone care?
3. **AI/Data Core** — What specific model(s), data pipeline(s), or AI technique(s) power
   the feature? Be concrete (e.g., "sentence-transformer embeddings + cosine
   similarity for matching" not just "uses AI to recommend things").
4. **Prototype Scope** — What would the Streamlit prototype actually show? What
   screens/interactions would exist?
5. **Wow Factor** — Why would this score 9-10? What makes it non-trivial and
   impressive?
6. **Feasibility on Streamlit Cloud** — Any risks or limitations? How would you handle
   them?
7. **Portfolio Extensibility** — How could Assignment 2 meaningfully improve on this?

## Selection Criteria (rank your proposals against these)
- **Originality:** Avoid cliché ideas (sentiment analysis dashboards, basic chatbots,
  generic recommendation engines). Surprise me.
- **Purposefulness:** The prototype should demonstrate HOW a user would interact with
  the AI feature in context — not just show model outputs.
- **Technical depth:** At least one non-trivial AI component that goes beyond a single
  API call.
- **Visual/UX potential:** Can this look impressive in Streamlit with creative use of
  widgets (tabs, columns, chat_input, st.map, custom components, feedback widgets,
  etc.)?
- **Deployability:** Must actually work on Streamlit Community Cloud.

## Important
- Think like a product designer, not just an engineer. The EXPERIENCE matters.
- Each idea should target a DIFFERENT domain or product category.
- Be bold — "mind-blowing" is literally in the rubric.