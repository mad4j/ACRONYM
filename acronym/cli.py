"""Command-line interface for ACRONYM.

Usage::

    # Train a model
    acronym train data/en/samples.json --lang en

    # Detect acronyms in a document
    acronym detect report.docx --lang en --format json

    # Show help
    acronym --help
    acronym train --help
    acronym detect --help
"""

import argparse
import json
import os
import sys

from .detector import detect_acronyms
from .trainer import DEFAULT_DATA_DIR, train_from_file


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    """Handle the ``train`` sub-command."""
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Error: training data file not found: {data_path}", file=sys.stderr)
        return 1

    try:
        train_from_file(data_path, lang=args.lang, model_dir=args.model_dir)
    except (ValueError, OSError) as exc:
        print(f"Error during training: {exc}", file=sys.stderr)
        return 1

    print(f"Training complete for language '{args.lang}'.")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    """Handle the ``detect`` sub-command."""
    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        results = detect_acronyms(
            args.input,
            lang=args.lang,
            model_dir=args.model_dir,
            threshold=args.threshold,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        if not results:
            print("No acronyms found.")
        else:
            print(f"Found {len(results)} acronym(s):\n")
            col1, col2 = 15, 10
            header = f"{'Acronym':<{col1}} {'Confidence':<{col2}} Definition"
            print(header)
            print("-" * max(70, len(header)))
            for r in results:
                print(
                    f"{r['acronym']:<{col1}} {r['confidence']:<{col2}.3f} {r['definition']}"
                )

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acronym",
        description=(
            "ACRONYM – Micro AI for extracting acronyms and definitions "
            "from .docx files."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -- train ----------------------------------------------------------------
    p_train = sub.add_parser(
        "train",
        help="Train a language-specific model from a labelled JSON dataset.",
    )
    p_train.add_argument("data", help="Path to the training data JSON file.")
    p_train.add_argument(
        "--lang",
        "-l",
        default="en",
        choices=["en", "it"],
        help="Language of the training data (default: en).",
    )
    p_train.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Directory where the trained model is saved (default: models/).",
    )

    # -- detect ---------------------------------------------------------------
    p_detect = sub.add_parser(
        "detect",
        help="Detect acronyms and definitions in a .docx document.",
    )
    p_detect.add_argument("input", help="Path to the .docx file to analyse.")
    p_detect.add_argument(
        "--lang",
        "-l",
        default="en",
        choices=["en", "it"],
        help="Language of the document (default: en).",
    )
    p_detect.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Directory containing trained models (default: models/).",
    )
    p_detect.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum confidence score [0-1] (default: 0.5).",
    )
    p_detect.add_argument(
        "--format",
        "-f",
        default="table",
        choices=["table", "json"],
        help="Output format (default: table).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        sys.exit(_cmd_train(args))
    elif args.command == "detect":
        sys.exit(_cmd_detect(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
