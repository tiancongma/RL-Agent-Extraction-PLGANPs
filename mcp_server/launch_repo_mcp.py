from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from mcp_server.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
