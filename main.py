"""Entry point for tiny-rlm M&A extraction."""

import argparse
import json
import sys

from tiny_rlm.config import load_config
from tiny_rlm.logging import LogLevel, RLMLogger
from tiny_rlm.mna.extractor import MnAExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract M&A transactions from documents using RLM"
    )
    parser.add_argument("input", help="Path to document text file, or '-' for stdin")
    parser.add_argument(
        "--config", default="rlm_config.yaml", help="Path to rlm_config.yaml"
    )
    parser.add_argument("--schema", default=None, help="Path to mna.yaml schema")
    parser.add_argument(
        "--raw", action="store_true", help="Output raw LLM result instead of parsed"
    )

    # Logging flags
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Rich debug panels with syntax-highlighted code",
    )
    log_group.add_argument(
        "--silent", action="store_true", help="Suppress all RLM logging output"
    )
    log_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress RLM logging (deprecated, use --silent)",
    )

    parser.add_argument(
        "--log-dir", default=None, help="Directory for JSONL log files (default: logs/)"
    )
    parser.add_argument(
        "--no-logs", action="store_true", help="Disable JSONL file logging"
    )
    args = parser.parse_args()

    # Read document text
    if args.input == "-":
        document_text = sys.stdin.read()
    else:
        with open(args.input) as f:
            document_text = f.read()

    if not document_text.strip():
        print("Error: empty document", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    # Determine log level
    if args.debug:
        level = LogLevel.DEBUG
    elif args.silent or args.quiet:
        level = LogLevel.QUIET
    else:
        level = LogLevel(config.log_level)

    log_dir = args.log_dir or config.log_dir
    enable_jsonl = not args.no_logs and config.enable_jsonl

    logger = RLMLogger(level=level, log_dir=log_dir, enable_jsonl=enable_jsonl)
    logger.start()

    try:
        extractor = MnAExtractor(
            config=config,
            schema_path=args.schema,
            verbose=level != LogLevel.QUIET,
            logger=logger,
        )

        result = extractor.extract(document_text)

        if args.raw:
            print(json.dumps(result.raw_result, indent=2, default=str))
        else:
            output = {
                "transactions": [
                    t.model_dump(exclude_none=True) for t in result.transactions
                ],
                "usage": result.usage,
            }
            print(json.dumps(output, indent=2, default=str))
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
