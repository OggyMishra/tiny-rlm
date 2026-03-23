"""M&A-specific system prompts built from the schema.

Combines RLM REPL instructions (from fast-rlm patterns) with M&A extraction
rules from schema/mna.yaml.
"""

from tiny_rlm.mna.schema import get_field_descriptions, get_field_groups, get_system_prompt


def build_mna_root_prompt(schema: dict) -> str:
    """Build the root agent system prompt for M&A extraction.

    Combines:
    - RLM REPL instructions (context, llm_query, FINAL, asyncio.gather)
    - M&A extraction rules from schema (verbatim)
    - Strategy guidance for chunked parallel extraction
    """
    mna_rules = get_system_prompt(schema)
    field_groups = get_field_groups(schema)
    field_descs = get_field_descriptions(schema)

    # Build field group summary for the LLM
    group_summary = _format_field_groups(field_groups, field_descs)

    return f"""\
You are an expert M&A data extraction agent operating in a REPL environment.

## REPL Environment

You have access to:

1. `context` — a variable holding the full document text. You never see it directly in your messages; use code to explore it.
2. `llm_query(prompt)` — an async function that calls a sub-LLM. Use `await llm_query(...)`. The return value is the actual Python object the sub-agent passed to FINAL (dict, list, string, etc). Do NOT wrap in eval() or json.loads().
3. `FINAL(x)` — call this to return your answer. Pass the actual variable: `FINAL(result)` not `FINAL("result")`.

Write Python code in a single ```repl block. Variables persist across turns (like Jupyter). NEVER delete `context`.

## M&A Extraction Rules

{mna_rules}

## Field Groups for Extraction

{group_summary}

## Strategy

1. **Inspect context**: Check type, length, first/last 500 chars to understand document structure.
2. **Identify deals**: Search for deal indicators (acquir, merg, acquis, transaction, purchase agreement). Each unique target = one transaction.
3. **Parallel extract by field group**: Use `asyncio.gather()` to extract field groups in parallel via `llm_query()`:
   ```python
   tasks = [
       llm_query(f"Extract DATES fields from this M&A document section:\\n{{relevant_text}}\\n\\nFields: {{date_field_descriptions}}\\nReturn a dict with field names as keys."),
       llm_query(f"Extract PARTICIPANTS fields..."),
       llm_query(f"Extract CONSIDERATION fields..."),
       llm_query(f"Extract FUNDING and ADVISOR fields..."),
   ]
   results = await asyncio.gather(*tasks)
   ```
4. **Aggregate**: Merge sub-results into transaction dicts.
5. **Verify**: For key fields (acquirer, target, consideration_size), verify values exist in original text.
6. **Return**: `FINAL(transactions)` where transactions is a `list[dict]`.

## Rules

- All code in a SINGLE ```repl block per turn.
- Outputs are truncated to last N characters — use `llm_query()` for analyzing large text.
- Multi-turn: inspect first, then extract, then verify, then FINAL.
- Maximize parallelism with `asyncio.gather()`.
- For multi-deal documents: extract each deal independently.
- Return `null` for fields not found — do NOT hallucinate values.
- Each transaction must have at minimum: a target name or acquirer name.
"""


def build_field_extraction_prompt(
    group_name: str,
    field_names: list[str],
    field_descs: dict[str, str],
) -> str:
    """Build a sub-agent prompt for extracting a specific field group.

    Used by the root agent when constructing llm_query() calls.
    """
    fields_text = "\n".join(
        f"- **{name}**: {field_descs.get(name, 'No description')}"
        for name in field_names
    )

    return f"""\
You are extracting {group_name} fields from an M&A document section.

The REPL has `context` containing the relevant text. Extract these fields:

{fields_text}

## Rules
- Extract values *verbatim* from the text. No paraphrasing.
- Return `null` for fields genuinely not found.
- For array fields, return `[]` if no values found.
- For entity objects (acquirors, target, sellers), extract all sub-fields (name, legal_name, entity_type, ticker, country, state_province).
- Inspect context first, then extract, then call `FINAL(result_dict)`.

Return a dict via `FINAL(extracted_fields)` where keys are field names.
"""


def _format_field_groups(
    groups: dict[str, list[str]],
    descs: dict[str, str],
) -> str:
    """Format field groups with descriptions for the system prompt."""
    parts = []
    for group_name, field_names in groups.items():
        if not field_names:
            continue
        label = group_name.upper().replace("_", " ")
        lines = [f"### {label} ({len(field_names)} fields)"]
        for name in field_names:
            desc = descs.get(name, "")
            # Truncate long descriptions
            short = desc[:120] + "..." if len(desc) > 120 else desc
            lines.append(f"- `{name}`: {short}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
