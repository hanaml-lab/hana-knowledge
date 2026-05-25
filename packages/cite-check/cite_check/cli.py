"""cite-check REPORT --sources DIR [--format md|html|json] [--revise]"""

from __future__ import annotations

import argparse
import sys

from cite_check.checker import Checker
from cite_check.revise import revise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cite-check", description=__doc__)
    ap.add_argument("report")
    ap.add_argument("--sources", required=True)
    ap.add_argument("--format", choices=["md", "html", "json"], default="md")
    ap.add_argument("--nli", default="lexical")
    ap.add_argument("--verifier", default=None)
    ap.add_argument("--reviewer", default=None)
    ap.add_argument("--revise", action="store_true", help="print the revised report instead of the review")
    args = ap.parse_args(argv)

    checker = Checker(nli=args.nli, verifier=args.verifier, reviewer=args.reviewer)
    review = checker.check(args.report, args.sources)
    if args.revise:
        sys.stdout.write(revise(review))
    else:
        sys.stdout.write({"md": review.to_markdown, "html": review.to_html, "json": review.to_json}[args.format]())
    return 1 if review.issues else 0


if __name__ == "__main__":
    sys.exit(main())
