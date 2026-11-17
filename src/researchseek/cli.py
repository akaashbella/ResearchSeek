"""Command-line entry point."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="researchseek",
        description="Local semantic and keyword search over research notes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Index research notes", description="Index research notes locally.")
    index.add_argument("directory", type=Path, help="Corpus root containing .txt and .md notes")
    index.add_argument("--db", type=Path, default=Path("researchseek.sqlite3"))
    index.add_argument("--rebuild", action="store_true", help="Replace corpus/configuration atomically")
    index.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    index.add_argument("--revision", help="Model commit/tag; defaults to the indexed commit or pinned initial revision")
    index.add_argument("--cache-dir", type=Path, default=Path(".tools/model-cache"))
    index.add_argument("--download-model", action="store_true", help="Explicitly allow model acquisition")
    index.add_argument("--max-tokens", type=_positive_int, default=200)
    index.add_argument("--overlap", type=_nonnegative_int, default=32)
    index.add_argument("--batch-size", type=_positive_int, default=32)
    index.add_argument("--json", action="store_true", help="Report counts and elapsed seconds as JSON")
    search = commands.add_parser("search", help="Search indexed passages", description="Search indexed passages.")
    search.add_argument("query", type=_query_text)
    search.add_argument("--db", type=Path, default=Path("researchseek.sqlite3"))
    search.add_argument("--mode", choices=("semantic", "keyword"), default="semantic")
    search.add_argument("--top-k", type=_positive_int, default=5)
    search.add_argument("--cache-dir", type=Path, default=Path(".tools/model-cache"))
    search.add_argument("--json", action="store_true", help="Output a JSON search response")
    for name, summary in (
        ("benchmark", "Evaluate retrieval correctness, speed, and quality"),
    ):
        commands.add_parser(
            name,
            help=f"{summary} (not implemented yet)",
            description=f"{summary}. Not implemented yet.",
        )

    args = parser.parse_args(argv)
    if args.command == "search":
        from researchseek.search import QueryError, search as search_passages

        try:
            results = search_passages(args.db, args.query, mode=args.mode, top_k=args.top_k,
                                     cache_dir=args.cache_dir)
            if args.json:
                print(json.dumps({"query": args.query, "mode": args.mode, "top_k": args.top_k,
                                  "results": [asdict(result) for result in results]},
                                 ensure_ascii=False, allow_nan=False))
            elif not results:
                print("No matches.")
            else:
                for result in results:
                    print(f"{result.rank}. {result.path}:{result.start_line}-{result.end_line} "
                          f"{result.score_type}={result.score:.6f}")
                    print(result.text)
                    print()
            return 0
        except QueryError as error:
            parser.error(str(error))
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            print(f"researchseek: {error}", file=sys.stderr)
            return 1
    if args.command == "index":
        started = perf_counter()
        try:
            from researchseek.chunking import ChunkConfig
            from researchseek.documents import discover_documents
            from researchseek.embeddings import DEFAULT_REVISION, SentenceTransformerEmbedder
            from researchseek.indexing import index_corpus
            from researchseek.storage import Storage

            discover_documents(args.directory, args.db)
            revision = args.revision
            if args.db.exists() and not args.rebuild:
                with Storage(args.db) as storage:
                    previous = storage.configuration()
                if previous is not None:
                    if previous.model_id != args.model:
                        raise ValueError("Incompatible model configuration; use --rebuild")
                    revision = revision or previous.model_revision
            embedder = SentenceTransformerEmbedder(args.model, revision=revision or DEFAULT_REVISION,
                                                   cache_dir=args.cache_dir, allow_download=args.download_model)
            capacity = min(args.max_tokens, embedder.tokenizer.max_tokens) - embedder.tokenizer.special_tokens
            if capacity < 1 or args.overlap >= capacity:
                parser.error("token limit must leave content space and overlap must be below that capacity")
            result = asdict(index_corpus(args.directory, args.db, embedder,
                                        config=ChunkConfig(args.max_tokens, args.overlap),
                                        rebuild=args.rebuild, batch_size=args.batch_size))
            result["elapsed_seconds"] = perf_counter() - started
            if args.json:
                print(json.dumps(result, sort_keys=True))
            else:
                print("Added: {added}  Updated: {updated}  Unchanged: {unchanged}  Deleted: {deleted}".format(**result))
                print("Documents: {documents}  Chunks: {chunks}  Elapsed: {elapsed_seconds:.3f}s".format(**result))
            return 0
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            print(f"researchseek: {error}", file=sys.stderr)
            return 1
    print(f"researchseek: {args.command} is not implemented yet.", file=sys.stderr)
    return 1


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _query_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("query must contain non-whitespace text")
    return value


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result
