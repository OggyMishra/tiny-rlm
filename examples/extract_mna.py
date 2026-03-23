"""Demo: Extract M&A transactions from a sample document."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_rlm.mna.extractor import MnAExtractor


SAMPLE_TEXT = """\
FOR IMMEDIATE RELEASE

Sunoco LP to Acquire Parkland Corporation in US$9.1 Billion Transaction

DALLAS and CALGARY, May 5, 2025 – Sunoco LP (NYSE: SUN) ("Sunoco") and Parkland
Corporation (TSX: PKI) ("Parkland") today announced that they have entered into a
definitive arrangement agreement under which Sunoco will acquire all of the outstanding
common shares of Parkland for consideration consisting of, at the election of each
Parkland shareholder, (i) 0.295 SUNCorp units and C$19.80 in cash, (ii) C$44.00 in
cash, or (iii) 0.536 SUNCorp units per Parkland share.

The transaction is valued at approximately US$9.1 billion, including the assumption of
Parkland's net debt. Based on the closing price of Sunoco common units on May 2, 2025,
the consideration represents a premium of approximately 25% to Parkland's undisturbed
share price.

The transaction is expected to close in the second half of 2025, subject to Parkland
shareholder approval, court approval, regulatory approvals and other customary closing
conditions.

Energy Transfer LP (NYSE: ET), Sunoco's parent company, will own a majority stake in
the combined entity.

Barclays and RBC Capital Markets are acting as financial advisors to Sunoco. Goldman
Sachs Canada Inc. and BofA Securities are acting as financial advisors to Parkland.
Stikeman Elliott LLP, Weil, Gotshal & Manges LLP, and Vinson & Elkins LLP are acting
as legal advisors to Sunoco.

The cash portion will be financed by a combination of cash and equity, with cash portion
financed by a $2.65 billion bridge term loan.
"""


def main():
    print("=" * 60)
    print("tiny-rlm M&A Extraction Demo")
    print("=" * 60)

    extractor = MnAExtractor(verbose=True)
    result = extractor.extract(SAMPLE_TEXT)

    print("\n" + "=" * 60)
    print(f"Extracted {len(result.transactions)} transaction(s)")
    print("=" * 60)

    for i, txn in enumerate(result.transactions):
        print(f"\n--- Transaction {i + 1} ---")
        data = txn.model_dump(exclude_none=True)
        print(json.dumps(data, indent=2, default=str))

    print(f"\n--- Usage ---")
    print(json.dumps(result.usage, indent=2))


if __name__ == "__main__":
    main()
