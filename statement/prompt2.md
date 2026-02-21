# PHASE 2 — BUILD: Full Streamlit Prototype

I've chosen idea [NUMBER]. Now build the complete, deployment-ready prototype.

## Deliverables (produce ALL of these)

### 1. Project Structure
Provide the full file/folder structure with clear separation of concerns:
- `app.py` — Main Streamlit app
- `model/` or `pipeline/` — Prediction logic separated from UI (as required)
- `utils/` — Helper functions
- `requirements.txt` — Pinned dependencies (Streamlit Cloud compatible)
- `README.md` — Setup and deployment instructions
- `.streamlit/config.toml` — Theme/config if needed

### 2. Streamlit App (`app.py`)
Build a complete, polished Streamlit application that:
- Has a clear visual hierarchy and professional layout
- Uses **at least 3 Streamlit widgets not commonly used in basic tutorials**
  (e.g., `st.chat_input`, `st.tabs`, `st.expander`, `st.metric`, `st.columns`,
  `st.feedback`, `st.popover`, `st.toggle`, custom components)
- Implements proper **state management** with `st.session_state`
- Handles errors and edge cases gracefully (loading states, empty inputs, API failures)
- Includes inline guidance so a first-time user understands what to do
- Feels like a REAL product feature, not a homework demo

### 3. AI/Model Pipeline
- Separate the model/prediction logic into its own module
- If using a pre-trained model: include the loading and inference code
- If using OpenAI/Anthropic API: implement proper prompt engineering with
  structured outputs
- If any component is mocked: make the mock realistic and document what
  a production version would use
- Include a brief code comment explaining what each AI component does

### 4. Data Pipeline
- If using external data: include data fetching, caching (`@st.cache_data`
  or `@st.cache_resource`), and any preprocessing
- If using synthetic/sample data: make it realistic and representative

### 5. Deployment Config
- `requirements.txt` with exact versions compatible with Streamlit Cloud
- Any secrets management via `st.secrets` (with instructions for setup)
- `.streamlit/config.toml` for any custom theming

## Code Quality Standards
- Type hints on all functions
- Docstrings on modules and key functions
- Clean separation: UI layer never contains business logic
- No hardcoded API keys — use `st.secrets` or environment variables
- Responsive layout that works on different screen sizes

## UX/Design Requirements
- Consistent color scheme and visual language
- Meaningful use of whitespace and layout columns
- Progress indicators for any async/slow operations
- Micro-interactions that make the app feel alive (success messages,
  transitions, dynamic updates)

## Produce the complete code for every file. Do not summarize or truncate.
## After the code, provide:
1. Step-by-step deployment instructions for Streamlit Community Cloud
2. A list of Streamlit widgets used (with justification for each)
3. A brief explanation of the AI pipeline architecture