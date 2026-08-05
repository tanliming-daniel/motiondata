from __future__ import annotations

import argparse
import json
from pathlib import Path

from cache_manifest import CacheManifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and prune the GLB cache.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/models"))
    parser.add_argument("--max-gb", type=float, default=None, help="Keep the cache under this size.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--cleanup-tmp", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Actually delete files; default is dry-run.")
    args = parser.parse_args()

    manifest = CacheManifest(args.cache_dir)
    try:
        if args.cleanup_tmp:
            manifest.cleanup_temporary()
        manifest.reconcile()
        removed = []
        if args.max_gb is not None:
            removed = manifest.prune(
                max_bytes=max(0, int(args.max_gb * 1024**3)),
                dataset=args.dataset,
                dry_run=not args.apply,
            )
        print(json.dumps({"stats": manifest.stats(), "dry_run": not args.apply, "would_remove": [str(p) for p in removed]}, indent=2))
    finally:
        manifest.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
