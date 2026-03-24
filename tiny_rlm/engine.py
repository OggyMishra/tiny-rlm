"""Core RLM loop — adopted from fast-rlm patterns.

Flow per document:
1. Init REPL with context, FINAL(), llm_query()
2. Step 0: auto-inspect context (type, length, first/last 500 chars)
3. Loop: LLM generates code → REPL executes → output fed back → repeat
4. On FINAL(): return result
"""

from __future__ import annotations

import re
import time
from typing import Any

from tiny_rlm.config import RLMConfig, UsageTracker
from tiny_rlm.llm import LLMClient
from tiny_rlm.logging.logger import RLMLogger
from tiny_rlm.logging.run_context import RunContext
from tiny_rlm.prompts import LEAF_SYSTEM_PROMPT, ROOT_SYSTEM_PROMPT, STEP0_INSPECT_CODE
from tiny_rlm.repl import REPLEnv

# ANSI helpers for legacy verbose logging (used when no logger provided)
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _log(verbose: bool, color: str, label: str, text: str) -> None:
    if verbose:
        preview = text[:500] + "..." if len(text) > 500 else text
        print(f"{color}{_BOLD}[{label}]{_RESET} {preview}\n")


def _extract_repl_block(text: str) -> str | None:
    """Extract code from a single ```repl block."""
    matches = re.findall(r"```repl\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return "\n".join(m.strip() for m in matches)
    return None


def _truncate_last_n(text: str, n: int) -> str:
    """Truncate showing LAST n chars (fast-rlm pattern)."""
    if not text:
        return "[EMPTY OUTPUT]"
    if len(text) > n:
        return f"[TRUNCATED: Last {n} chars shown].. " + text[-n:]
    return "[FULL OUTPUT SHOWN]... " + text


async def _run_subagent(
    context: str,
    depth: int,
    config: RLMConfig,
    tracker: UsageTracker,
    verbose: bool,
    system_prompt_override: str | None = None,
    parent_ctx: RunContext | None = None,
    logger: RLMLogger | None = None,
) -> Any:
    """Run an RLM sub-agent at the given depth. Called by llm_query() inside REPL."""

    agent_start = time.time()
    is_leaf = depth >= config.max_depth
    model_name = config.sub_agent
    system_prompt = system_prompt_override or (
        LEAF_SYSTEM_PROMPT if is_leaf else ROOT_SYSTEM_PROMPT
    )

    # Create run context for this sub-agent
    ctx: RunContext | None = None
    if parent_ctx is not None:
        ctx = parent_ctx.create_child(model=model_name, is_leaf=is_leaf)
    elif logger is not None:
        ctx = RunContext.create_root(model=model_name, is_leaf=is_leaf)

    if logger and ctx:
        logger.log_agent_start(ctx, context[:200])

    llm = LLMClient(model=model_name)

    # Build llm_query for this depth (None if leaf)
    llm_query_fn = None
    if not is_leaf:

        async def llm_query_fn(sub_ctx: str) -> Any:
            child_ctx = (
                ctx.create_child(
                    model=config.sub_agent, is_leaf=(depth + 1) >= config.max_depth
                )
                if ctx
                else None
            )
            if logger and ctx and child_ctx:
                logger.log_subagent_spawn(ctx, child_ctx, sub_ctx[:200])
            return await _run_subagent(
                sub_ctx,
                depth + 1,
                config,
                tracker,
                verbose,
                system_prompt_override=None,
                parent_ctx=ctx,
                logger=logger,
            )

    repl = REPLEnv(llm_query_fn=llm_query_fn)
    repl.load_context(context)

    # Step 0: auto-inspect context
    step0_result = repl.execute(STEP0_INSPECT_CODE)
    step0_output = step0_result.stdout + step0_result.stderr
    if not logger:
        _log(verbose, _DIM, f"SUB-AGENT d={depth} step0", step0_output)

    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                f"Outputs will be truncated to last {config.truncate_len} characters.\n"
                f"code:\n```repl\n{STEP0_INSPECT_CODE}\n```\n"
                f"Output:\n{step0_output.strip()}"
            ),
        }
    ]

    iterations_used = 0
    for i in range(config.max_calls_per_subagent):
        tracker.check_budget(config)
        iterations_used = i + 1

        # LLM call
        call_start = time.time()
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response, usage = llm.complete_with_usage(full_messages)
        call_latency = time.time() - call_start
        tracker.track(usage)

        if logger and ctx:
            logger.log_llm_call(ctx, i, usage, call_latency)
        else:
            _log(verbose, _CYAN, f"SUB d={depth} iter={i}", response or "")

        code = _extract_repl_block(response or "")

        if logger and ctx:
            logger.log_code_extract(ctx, i, code)

        if not code:
            messages.append(
                {
                    "role": "assistant",
                    "content": response or "",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Error: could not extract code — use a ```repl block.",
                }
            )
            continue

        # Execute
        result = await repl.execute_async(code)
        output = result.stdout + result.stderr
        truncated = _truncate_last_n(output, config.truncate_len)

        if logger and ctx:
            logger.log_repl_execution(
                ctx,
                i,
                result.stdout,
                result.stderr,
                result.execution_time,
                bool(result.stderr),
            )
        else:
            _log(verbose, _YELLOW, f"SUB d={depth} output", truncated)

        if repl.final_set:
            if logger and ctx:
                logger.log_final_call(ctx, i, str(repl.final_result))
                logger.log_agent_end(
                    ctx, iterations_used, time.time() - agent_start, success=True
                )
            else:
                _log(
                    verbose,
                    _GREEN,
                    f"SUB d={depth} FINAL",
                    str(repl.final_result)[:200],
                )
            return repl.final_result

        messages.append({"role": "assistant", "content": response or ""})
        messages.append({"role": "user", "content": f"Output:\n{truncated}"})

    if logger and ctx:
        logger.log_agent_end(
            ctx, iterations_used, time.time() - agent_start, success=False
        )
    raise RuntimeError(
        f"Sub-agent at depth={depth} exhausted {config.max_calls_per_subagent} iterations without FINAL()"
    )


class RLMEngine:
    """Recursive Language Model engine.

    Usage:
        engine = RLMEngine(config)
        result = asyncio.run(engine.run(context="...", query="..."))
    """

    def __init__(
        self,
        config: RLMConfig | None = None,
        verbose: bool = True,
        logger: RLMLogger | None = None,
    ):
        self.config = config or RLMConfig()
        self.verbose = verbose
        self.logger = logger
        self.tracker = UsageTracker()

    async def run(
        self,
        context: str | dict | list,
        query: str,
        system_prompt_override: str | None = None,
    ) -> Any:
        """Run the RLM loop on the given context and query."""

        run_start = time.time()
        self.tracker = UsageTracker()  # Fresh tracker per run
        config = self.config
        logger = self.logger
        is_leaf = config.max_depth <= 0
        system_prompt = system_prompt_override or (
            LEAF_SYSTEM_PROMPT if is_leaf else ROOT_SYSTEM_PROMPT
        )

        root_llm = LLMClient(model=config.primary_agent)

        # Create root run context
        root_ctx: RunContext | None = None
        if logger:
            root_ctx = RunContext.create_root(
                model=config.primary_agent, is_leaf=is_leaf
            )
            logger.log_agent_start(root_ctx, query)

        # Build llm_query for root (depth=0)
        llm_query_fn = None
        if not is_leaf:

            async def llm_query_fn(ctx: str) -> Any:
                child_ctx = (
                    root_ctx.create_child(
                        model=config.sub_agent,
                        is_leaf=1 >= config.max_depth,
                    )
                    if root_ctx
                    else None
                )
                if logger and root_ctx and child_ctx:
                    logger.log_subagent_spawn(root_ctx, child_ctx, ctx[:200])
                return await _run_subagent(
                    ctx,
                    depth=1,
                    config=config,
                    tracker=self.tracker,
                    verbose=self.verbose,
                    system_prompt_override=None,
                    parent_ctx=root_ctx,
                    logger=logger,
                )

        repl = REPLEnv(llm_query_fn=llm_query_fn)

        # Load context — convert to string if needed
        if isinstance(context, (dict, list)):
            import json

            repl.load_context(json.dumps(context, indent=2))
        else:
            repl.load_context(context)

        if not logger:
            _log(self.verbose, _GREEN, "INIT", f"Context loaded. Query: {query}")

        # Step 0: auto-inspect
        step0_result = repl.execute(STEP0_INSPECT_CODE)
        step0_output = step0_result.stdout + step0_result.stderr
        if not logger:
            _log(self.verbose, _DIM, "STEP 0", step0_output)

        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"Outputs will be truncated to last {config.truncate_len} characters.\n"
                    f"code:\n```repl\n{STEP0_INSPECT_CODE}\n```\n"
                    f"Output:\n{step0_output.strip()}"
                ),
            }
        ]

        # Main RLM loop
        iterations_used = 0
        for i in range(config.max_calls_per_subagent):
            self.tracker.check_budget(config)
            iterations_used = i + 1

            # Build user prompt with query
            if i == 0:
                user_hint = (
                    f"You have NOT seen the context yet (only metadata above). "
                    f"Your first step MUST be to inspect it. Do NOT provide a final answer yet.\n\n"
                    f'Answer this query: "{query}"\n'
                    f"Write a ```repl block to explore the context."
                )
            else:
                user_hint = (
                    f"Continue exploring or provide your final answer.\n"
                    f'Query: "{query}"\n'
                    f"Write a ```repl block, or call FINAL(answer) if ready."
                )
            messages.append({"role": "user", "content": user_hint})

            # LLM call
            call_start = time.time()
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response, usage = root_llm.complete_with_usage(full_messages)
            call_latency = time.time() - call_start
            self.tracker.track(usage)

            if logger and root_ctx:
                logger.log_llm_call(root_ctx, i, usage, call_latency)
            else:
                _log(self.verbose, _CYAN, f"ROOT iter={i}", response or "")

            code = _extract_repl_block(response or "")

            if logger and root_ctx:
                logger.log_code_extract(root_ctx, i, code)

            if not code:
                messages.append({"role": "assistant", "content": response or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "Error: could not extract code — use a ```repl block.",
                    }
                )
                continue

            # Execute in REPL
            result = await repl.execute_async(code)
            output = result.stdout + result.stderr
            truncated = _truncate_last_n(output, config.truncate_len)

            if logger and root_ctx:
                logger.log_repl_execution(
                    root_ctx,
                    i,
                    result.stdout,
                    result.stderr,
                    result.execution_time,
                    bool(result.stderr),
                )
            else:
                _log(self.verbose, _YELLOW, f"ROOT output iter={i}", truncated)

            if repl.final_set:
                if logger and root_ctx:
                    logger.log_final_call(root_ctx, i, str(repl.final_result))
                    logger.log_agent_end(
                        root_ctx, iterations_used, time.time() - run_start, success=True
                    )
                else:
                    _log(
                        self.verbose,
                        _GREEN,
                        "DONE",
                        str(repl.final_result)[:500],
                    )
                return repl.final_result

            messages.append({"role": "assistant", "content": response or ""})
            messages.append({"role": "user", "content": f"Output:\n{truncated}"})

        # Exhausted iterations — force final
        if not logger:
            _log(self.verbose, _RED, "MAX ITERS", "Forcing final answer")
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum iterations. "
                    "Provide your best answer NOW using FINAL(answer)."
                ),
            }
        )
        call_start = time.time()
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response, usage = root_llm.complete_with_usage(full_messages)
        call_latency = time.time() - call_start
        self.tracker.track(usage)
        iterations_used += 1

        if logger and root_ctx:
            logger.log_llm_call(root_ctx, iterations_used - 1, usage, call_latency)

        code = _extract_repl_block(response or "")
        if code:
            await repl.execute_async(code)
            if repl.final_set:
                if logger and root_ctx:
                    logger.log_final_call(
                        root_ctx, iterations_used - 1, str(repl.final_result)
                    )
                    logger.log_agent_end(
                        root_ctx, iterations_used, time.time() - run_start, success=True
                    )
                return repl.final_result

        if logger and root_ctx:
            logger.log_agent_end(
                root_ctx, iterations_used, time.time() - run_start, success=False
            )
        return response  # Last resort: raw LLM text
