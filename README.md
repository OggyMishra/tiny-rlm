# tiny-rlm

A Recursive Language Model (RLM) engine for extracting structured M&A transaction data from financial documents.

tiny-rlm uses a multi-turn LLM-REPL loop where the model generates Python code, executes it in a sandboxed environment, observes the output, and iterates — recursively spawning sub-agents for complex tasks. This approach allows the LLM to programmatically explore, parse, and extract from large unstructured documents rather than relying on a single prompt-response pass.

## How It Works

```
Document Text
     │
     ▼
┌─────────────────────────────────────────────┐
│  RLM Engine (Root Agent)                    │
│                                             │
│  1. Load document as `context` variable     │
│  2. Auto-inspect: type, length, preview     │
│  3. LLM generates Python in ```repl block   │
│  4. Sandboxed REPL executes code            │
│  5. Output fed back to LLM                  │
│  6. Repeat until FINAL(result) is called    │
│                                             │
│  ┌────────────────────────────────────┐     │
│  │  Sub-Agent (depth=1)              │     │
│  │  └─ Sub-Agent (depth=2, leaf)     │     │
│  │     No further recursion          │     │
│  └────────────────────────────────────┘     │
└─────────────────────────────────────────────┘
     │
     ▼
Structured JSON (validated MnATransaction)
```

Root and branch agents can call `await llm_query(prompt)` to spawn sub-agents for parallel field extraction via `asyncio.gather()`. Leaf agents (at max depth) work with Python string/regex operations only.

## Features

- **Recursive multi-agent architecture** — configurable depth for hierarchical document analysis
- **Sandboxed Python REPL** — safe execution with blocked builtins (`eval`, `exec`, `input`)
- **Budget guards** — token and cost limits to prevent runaway spending
- **Multi-model support** — any model via [LiteLLM](https://github.com/BerriAI/litellm) (Gemini, GPT, Claude, etc.)
- **86-field M&A schema** — comprehensive extraction covering dates, parties, consideration, funding, advisors, and termination fees
- **Pydantic validation** — structured output with typed models
- **Evaluation toolkit** — per-field F1, precision, recall, hallucination rate, and miss rate against a golden dataset

## Installation

```bash
# Clone and install
git clone <repo-url>
cd tiny-rlm
pip install -e ".[dev]"

# Configure LLM access
cp .env.example .env
# Edit .env with your LiteLLM proxy or API credentials:
#   LITELLM_BASE_URL=http://your-litellm-proxy
#   LITELLM_API_KEY=your-key
```

Requires Python 3.10+.

## Quick Start

### CLI

```bash
# Extract from a text file
python main.py document.txt

# Read from stdin
cat document.txt | python main.py -

# Quiet mode (suppress verbose logging)
python main.py document.txt --quiet

# Raw LLM output (skip Pydantic parsing)
python main.py document.txt --raw

# Custom config and schema
python main.py document.txt --config my_config.yaml --schema schema/mna.yaml
```

Output is JSON:

```json
{
  "transactions": [
    {
      "deal_type": "Acquisition",
      "deal_status": "Pending",
      "acquirors": [{"name": "Sunoco LP", "ticker": "SUN"}],
      "target": {"name": "Parkland Corporation", "ticker": "PKI"},
      "consideration_size": "US$9.1 billion",
      "deal_announcement_date": "May 5, 2025"
    }
  ],
  "usage": {
    "prompt_tokens": 12500,
    "completion_tokens": 3200,
    "total_tokens": 15700,
    "cost": 0.0042
  }
}
```

### Python API

```python
from tiny_rlm.mna.extractor import MnAExtractor

extractor = MnAExtractor(verbose=True)
result = extractor.extract(document_text)

for txn in result.transactions:
    print(txn.deal_type, txn.consideration_size)

print(result.usage)
```

## Configuration

### rlm_config.yaml

```yaml
# LLM models (LiteLLM format or proxy model group names)
primary_agent: "as-agent-gemini-3-flash"
sub_agent: "as-agent-gemini-3-flash"

# Recursion and iteration limits
max_depth: 2                    # Max agent recursion depth
max_calls_per_subagent: 15      # Max LLM calls per agent
truncate_len: 5000              # Truncate REPL output to last N chars

# Budget guards
max_money_spent: 5.0            # USD limit per run
max_completion_tokens: 200000   # Output token limit
max_prompt_tokens: 500000       # Input token limit
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LITELLM_BASE_URL` | LiteLLM proxy URL (optional — omit for direct API calls) |
| `LITELLM_API_KEY` | API key for proxy or direct provider |

## Project Structure

```
tiny-rlm/
├── main.py                     # CLI entry point
├── rlm_config.yaml             # Runtime configuration
├── pyproject.toml              # Package metadata & dependencies
│
├── tiny_rlm/                   # Core package
│   ├── config.py               # RLMConfig dataclass + UsageTracker
│   ├── engine.py               # RLM loop: LLM ↔ REPL ↔ sub-agents
│   ├── llm.py                  # LiteLLM client wrapper
│   ├── prompts.py              # Root/leaf system prompts + step-0 code
│   ├── repl.py                 # Sandboxed Python REPL (FINAL, llm_query)
│   │
│   └── mna/                    # M&A extraction specialization
│       ├── extractor.py        # Orchestrator: config → engine → parse
│       ├── models.py           # Pydantic models (MnATransaction, etc.)
│       ├── prompts.py          # M&A-specific prompt builders
│       └── schema.py           # YAML schema loader & field grouping
│
├── schema/
│   └── mna.yaml                # 86-field M&A extraction schema
│
├── examples/
│   ├── extract_mna.py          # Demo extraction on sample press release
│   └── evaluate.py             # Evaluation against golden dataset
│
└── docs/
    └── golden_dataset_with_content.csv  # Golden test dataset
```

## M&A Schema Fields

The schema defines 86 fields organized into groups:

| Group | Fields | Examples |
|-------|--------|---------|
| **Dates** | 10 | announcement date, signing date, completion date, rumor date |
| **Deal Features** | 5 | deal type, status, attitude, summary, rationale |
| **Participants** | 5 | acquirors, target, sellers, ultimate parents |
| **Consideration** | 30 | price per share, shares acquired %, cash/stock mix, debt assumed, valuation multiples |
| **Funding** | 9 | cash on hand, bridge loans, term loans, senior notes, equity placements |
| **Advisors** | 6 | financial advisors, legal advisors, auditors (both sides) |
| **Termination** | 2 | acquiror break-up fee, seller break-up fee |

See [schema/mna.yaml](schema/mna.yaml) for full field definitions, types, and extraction rules.

## Evaluation

Compare extraction results against the golden dataset:

```bash
# Run extraction on your documents, save as JSON:
#   { "doc_id_1": { ...transaction fields... }, "doc_id_2": { ... } }

python examples/evaluate.py extractions.json
```

Outputs per-field metrics with critical fields (deal_type, deal_status, acquirer_name, target_name, consideration_size, deal_announcement_date) weighted 3x in the overall score.

```
Field                                              F1   Prec    Rec  Halluc   Miss
-------------------------------------------------------------------------------------
consideration_size                              0.850  0.900  0.810   0.050  0.100 ***
deal_type                                       0.920  0.950  0.890   0.020  0.060 ***
...
Overall weighted F1:                            0.8742
(*** = critical field, 3x weight)
```

## Extending

To adapt tiny-rlm for a different extraction domain:

1. Create a new YAML schema in `schema/` (see [schema/README.md](schema/README.md))
2. Add domain-specific models in a new subpackage under `tiny_rlm/`
3. Build custom prompts with extraction rules for your domain
4. Reuse `RLMEngine` directly — it is domain-agnostic

## Dependencies

| Package | Purpose |
|---------|---------|
| [litellm](https://github.com/BerriAI/litellm) | Unified LLM API (100+ providers) |
| [pydantic](https://docs.pydantic.dev/) | Data validation & typed models |
| [pyyaml](https://pyyaml.org/) | Schema configuration |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable loading |

## License

See LICENSE file for details.
