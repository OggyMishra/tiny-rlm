"""Entry point for tiny-rlm M&A extraction."""

import argparse
import json
import sys

from tiny_rlm.config import load_config
from tiny_rlm.mna.extractor import MnAExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract M&A transactions from documents using RLM")
    parser.add_argument("input", help="Path to document text file, or '-' for stdin")
    parser.add_argument("--config", default="rlm_config.yaml", help="Path to rlm_config.yaml")
    parser.add_argument("--schema", default=None, help="Path to mna.yaml schema")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose RLM logging")
    parser.add_argument("--raw", action="store_true", help="Output raw LLM result instead of parsed")
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
    extractor = MnAExtractor(
        config=config,
        schema_path=args.schema,
        verbose=not args.quiet,
    )

    result = extractor.extract(document_text)

    if args.raw:
        print(json.dumps(result.raw_result, indent=2, default=str))
    else:
        output = {
            "transactions": [t.model_dump(exclude_none=True) for t in result.transactions],
            "usage": result.usage,
        }
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
