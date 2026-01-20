# LLM-DS
domain-flexible autonomous data analysis agent leveraging large language models

Status
- Pending — work in progress. The project is not complete and requires further implementation, testing, and integration.

What this repo contains (high level)
- data_cleaning: level-1/2 metrics, plan generation, execution agent
- data_analysis: profile → text, analysis thinker, plan generator, evaluation pipeline
- orchestrator: simple agent and optional state graph to drive ingest → clean → analyze
- utils: LLM clients (Groq), helpers
- Integration points for Gemini (google.generativeai) and GROQ — several modules expect API keys and a .env file

Minimal requirements (developer)
- Python 3.10+
- .env with required API keys (e.g., GEMINI_API_KEY, GROQ_API_KEY) for LLM calls
- Typical dependencies: pandas, python-dotenv, google-generativeai, groq (see project for exact imports)

Current notes
- Many modules include TODOs, placeholders, and developer test runners. The system orchestrates planning (LLM), plan execution, and analysis steps, but the full pipeline and error handling are not finalized.
- This README intentionally minimal — the project remains pending and needs completion before production use.

Next steps (high level)
- Finish implementation of missing logic and integration tests
- Add clear developer setup and run instructions once components are stable
- Harden error handling and safe execution of LLM-generated code
