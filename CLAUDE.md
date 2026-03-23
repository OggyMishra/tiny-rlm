# tiny-rlm

Recursive Language Model (RLM) engine for extracting structured M&A transaction data from financial documents using a multi-turn LLM-REPL loop.

## Project Overview

- **Language**: Python 3.10+
- **Package manager**: uv
- **Virtual env**: `.venv/` (Python 3.11)
- **Dependencies**: litellm, pydantic, pyyaml, python-dotenv
- **Dev dependencies**: pytest, ruff

## Quick Commands

```bash
# Install
uv sync --all-extras

# Run extraction
uv run python main.py document.txt
uv run python main.py document.txt --quiet --raw

# Lint
uv run ruff check .
uv run ruff format .

# Tests
uv run pytest

# Add a dependency
uv add <package>
uv add --dev <package>
```

## Architecture

The system uses a recursive multi-agent approach: a root LLM agent generates Python code, executes it in a sandboxed REPL, observes output, and iterates. Root/branch agents can spawn sub-agents via `llm_query()` for parallel field extraction using `asyncio.gather()`. Leaf agents (at max depth) use Python string/regex only.

### Core Loop (engine.py)
1. Load document as `context` variable in REPL
2. Step 0: auto-inspect context (type, length, preview)
3. LLM generates Python code in ` ```repl ` blocks
4. Sandboxed REPL executes code
5. Output fed back to LLM (truncated to `truncate_len` chars)
6. Repeat until `FINAL(result)` is called

### Key Files

```
main.py                        # CLI entry point (argparse)
rlm_config.yaml                # Runtime config (models, depth, budgets)
pyproject.toml                 # Package metadata

tiny_rlm/
  __init__.py                  # Exports: RLMConfig, RLMEngine, load_config
  config.py                    # RLMConfig dataclass, UsageTracker, load_config()
  engine.py                    # RLMEngine class + _run_subagent() — core RLM loop
  llm.py                       # LLMClient — LiteLLM wrapper with proxy support
  prompts.py                   # ROOT_SYSTEM_PROMPT, LEAF_SYSTEM_PROMPT, STEP0_INSPECT_CODE
  repl.py                      # REPLEnv — sandboxed exec with FINAL(), llm_query(), execute_async()

  mna/                         # M&A extraction specialization
    __init__.py                # Exports: MnAExtractor
    extractor.py               # MnAExtractor — orchestrator: config -> engine -> parse
    models.py                  # Pydantic models: MnATransaction (86 fields), ExtractionResult, parse_extraction()
    prompts.py                 # build_mna_root_prompt(), build_field_extraction_prompt()
    schema.py                  # YAML schema loader: load_schema(), get_field_groups(), get_field_descriptions()

schema/
  mna.yaml                     # 86-field M&A extraction schema (dates, participants, consideration, funding, advisors, termination)

examples/
  extract_mna.py               # Demo extraction on sample Sunoco/Parkland press release
  evaluate.py                  # Evaluation: per-field F1, precision, recall, hallucination rate, miss rate

docs/
  golden_dataset_with_content.csv  # Golden test dataset for evaluation
```

## Configuration

### rlm_config.yaml
- `primary_agent` / `sub_agent`: LiteLLM model names (or proxy model group names)
- `max_depth`: Max agent recursion depth (default: 2)
- `max_calls_per_subagent`: Max LLM iterations per agent (default: 15)
- `truncate_len`: Truncate REPL output to last N chars (default: 5000)
- `max_money_spent` / `max_completion_tokens` / `max_prompt_tokens`: Budget guards

### Environment Variables (.env)
- `LITELLM_BASE_URL`: LiteLLM proxy URL (optional — omit for direct API calls)
- `LITELLM_API_KEY`: API key for proxy or direct provider

## Key Patterns

- **LLM routing**: When `LITELLM_BASE_URL` is set, model names get `openai/` prefix for proxy routing (see `llm.py`)
- **Sandboxing**: REPL blocks `eval`, `exec`, `compile`, `input`, `globals`, `locals`, `breakpoint` builtins
- **Async execution**: Code containing `await` is wrapped in an async function via `execute_async()` in repl.py
- **Output parsing**: `parse_extraction()` in models.py handles LLM output normalization — flattens metadata wrappers (`raw_value`, `confidence_score`, `citations`), coerces entity fields, handles grouped/nested structures
- **Schema-driven prompts**: mna/prompts.py builds system prompts from mna.yaml schema, including field groups and descriptions

## Coding Conventions

- Use Python 3.10+ syntax (`str | None` union types, not `Optional[str]`)
- Pydantic v2 models with `model_validate()`, `model_dump()`
- Dataclasses for config (`RLMConfig`, `Usage`)
- Type hints on all function signatures
- No docstring required for trivial functions; keep descriptions concise
- Use `ruff` for linting and formatting

## Schema (mna.yaml)

86 fields organized into 7 groups:
- **Dates** (10): announcement, signing, completion, rumor, bid, LOI, revision, cancellation, expected timeframe, undisturbed price date
- **Deal Features** (5): type, status, attitude, summary, rationale
- **Participants** (5): acquirors[], acquiror_ultimate_parents[], target{}, sellers[], seller_ultimate_parents[]
- **Consideration** (30): price per share, shares acquired %, cash/stock mix, debt assumed, valuation multiples (EV/Revenue, EV/EBITDA, P/E, P/B)
- **Funding** (9): cash on hand, bridge loans, term loans, senior/subordinated/convertible notes, equity placements
- **Advisors** (6): financial advisors, legal advisors, auditors (both sides)
- **Termination** (2): acquiror break-up fee, seller break-up fee

## Important Notes

- The `.env` file contains API credentials — never commit it (it's in `.gitignore`)
- The golden dataset CSV is in `docs/` — used only for evaluation, not runtime
- Critical evaluation fields (3x weight): deal_type, deal_status, acquirer_name, target_name, consideration_size, deal_announcement_date
