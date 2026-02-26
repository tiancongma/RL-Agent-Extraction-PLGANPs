#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src" / "utils" / "paths.py").exists():
            sys.path.insert(0, str(p))
            return


_bootstrap_import_paths()
from src.utils import paths  # noqa: E402
from src.utils.run_id import is_valid_run_id  # noqa: E402


DEFAULT_RUN_ID = "run_20260219_1623_780eb83_goren18_weaklabels_v1"
DEFAULT_KEYS = [
    "L3H2RS2H",
    "7ZS858NS",
    "QLYKLPKT",
    "BB3JUVW7",
    "5GIF3D8W",
    "KBW3XWTT",
    "YGA8VQKU",
    "WIVUCMYG",
    "UFXX9WXE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug paper-local table registry candidate discovery.")
    p.add_argument("--keys", default="", help="Comma-separated keys or path to text file with one key per line.")
    p.add_argument("--run-id", default="", help="Required deterministic run_id from preflight.")
    p.add_argument("--out-subdir", default="", help="Required subdirectory under data/results/<run_id>/ for this run variant.")
    p.add_argument(
        "--tables-root",
        default=str(paths.DATA_CLEANED_DIR / "content_goren_2025" / "tables"),
        help="Per-key tables root. Expected structure: <tables_root>/<zotero_key>/",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print audit preview: exact per-key scanned directory and first 10 CSV filenames.",
    )
    return p.parse_args()


def _sanitize_out_subdir(s: str) -> str:
    v = str(s or "").strip().replace("\\", "/")
    if not v:
        raise ValueError(
            "ERROR: --out-subdir is required when reusing a run_id. Use a stage/variant folder name, e.g. stage2_validation or stage5_signature_iter001."
        )
    if Path(v).is_absolute():
        raise ValueError("ERROR: --out-subdir must be a relative path.")
    parts = [p for p in v.split("/") if p]
    if not parts or any(p == ".." for p in parts):
        raise ValueError("ERROR: --out-subdir cannot contain path traversal ('..').")
    return "/".join(parts)


def parse_keys_arg(keys_arg: str) -> list[str]:
    if not keys_arg.strip():
        return DEFAULT_KEYS
    kp = Path(keys_arg)
    if kp.exists() and kp.is_file():
        out: list[str] = []
        for line in kp.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s:
                out.append(s)
        return out if out else DEFAULT_KEYS
    parts = [x.strip() for x in keys_arg.split(",")]
    keys = [x for x in parts if x]
    return keys if keys else DEFAULT_KEYS


def read_key2txt(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        # Header format.
        if "key" in df.columns and "txt_path" in df.columns:
            for _, r in df.iterrows():
                k = str(r.get("key", "")).strip()
                rel = str(r.get("txt_path", "")).strip()
                if not k or not rel:
                    continue
                p = (path.parent / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
                out[k] = str(p)
            return out
    except Exception:
        pass
    # No-header fallback.
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            k = parts[0].strip()
            rel = parts[1].strip()
            if not k or not rel:
                continue
            p = (paths.PROJECT_ROOT / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
            out[k] = str(p)
    except Exception:
        return {}
    return out


def discover_key2txt_files() -> list[Path]:
    preferred = [
        paths.DATA_CLEANED_DIR / "content_goren_2025" / "key2txt.tsv",
        paths.DATA_CLEANED_DIR / "content" / "key2txt.tsv",
    ]
    all_found = sorted(paths.DATA_CLEANED_DIR.rglob("key2txt*.tsv"))
    ordered: list[Path] = []
    seen: set[str] = set()
    for p in preferred + all_found:
        ps = str(p.resolve())
        if ps in seen or not p.exists():
            continue
        seen.add(ps)
        ordered.append(p.resolve())
    return ordered


def _list_per_key_tables(
    *,
    tables_root: Path,
    zotero_key: str,
) -> tuple[Path, int, list[str], str, str]:
    key_dir = (tables_root / zotero_key).resolve()
    if not key_dir.exists() or not key_dir.is_dir():
        return key_dir, 0, [], "missing_key_dir", "per_key_dir_missing"

    manifest_path = key_dir / "tables_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
            names: list[str] = []
            selected = payload.get("selected_table_files", [])
            if isinstance(selected, list):
                names.extend([Path(str(x)).name for x in selected if str(x).strip()])
            if not names:
                for rec in payload.get("tables", []) if isinstance(payload.get("tables", []), list) else []:
                    if not isinstance(rec, dict):
                        continue
                    cand = str(rec.get("filename", "") or rec.get("csv_path", "")).strip()
                    if cand:
                        names.append(Path(cand).name)
            existing = sorted(set([n for n in names if (key_dir / n).exists()]))
            if existing:
                return key_dir, len(existing), existing[:10], "manifest", ""
            return key_dir, 0, [], "manifest", "manifest_present_but_no_tables"
        except Exception as e:
            # Bug fix: fallback still stays inside <tables_root>/<key> only.
            files = sorted([p.name for p in key_dir.glob("*.csv") if p.is_file()])
            reason = "" if files else f"manifest_parse_failed:{e}"
            return key_dir, len(files), files[:10], "fallback_glob_in_key_dir", reason

    files = sorted([p.name for p in key_dir.glob("*.csv") if p.is_file()])
    reason = "" if files else "no_csv_in_key_dir"
    return key_dir, len(files), files[:10], "glob_in_key_dir", reason


def main() -> None:
    args = parse_args()
    keys = parse_keys_arg(args.keys)
    run_id = str(args.run_id or "").strip()
    if not run_id:
        raise ValueError(
            "ERROR: --run-id is required. Generate/reuse a run_id via: python -m src.utils.run_preflight ..."
        )
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid --run-id (must match required regex): {run_id}")
    out_subdir = _sanitize_out_subdir(args.out_subdir)
    out_dir = paths.DATA_RESULTS_DIR / run_id / out_subdir / "audit_pack"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "paper_local_tables_debug_v1.tsv"
    tables_root = Path(args.tables_root).resolve()

    key2txt_files = discover_key2txt_files()
    key2txt_maps: list[tuple[Path, dict[str, str]]] = [(p, read_key2txt(p)) for p in key2txt_files]

    rows: list[dict[str, Any]] = []
    for key in keys:
        k = key.strip()
        resolved_text_path = ""
        key2txt_used = ""
        for p, km in key2txt_maps:
            if k in km:
                resolved_text_path = km[k]
                key2txt_used = str(p)
                break
        text_exists = bool(resolved_text_path and Path(resolved_text_path).exists())

        scanned_dir, total_csv, sample_names, scan_mode, scan_reason = _list_per_key_tables(
            tables_root=tables_root,
            zotero_key=k,
        )
        reason = scan_reason
        if not resolved_text_path and not reason:
            reason = "key2txt_missing"
        elif not text_exists and not reason:
            reason = "text_path_missing_on_disk"

        rows.append(
            {
                "zotero_key": k,
                "key2txt_used": key2txt_used,
                "resolved_text_path": resolved_text_path,
                "text_path_exists": text_exists,
                "candidate_tables_dirs": str(scanned_dir),
                "candidate_dir_csv_counts": f"{scanned_dir}:{int(total_csv)}",
                "total_n_csv_found": int(total_csv),
                "scan_mode": scan_mode,
                "sample_table_filenames": " | ".join(sample_names),
                "reason_if_zero": reason,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_tsv, sep="\t", index=False)

    # Console required prints.
    print(f"output_tsv={out_tsv}")
    reason_counts = (
        out_df["reason_if_zero"].astype(str).replace("", "(non_zero)").value_counts().rename_axis("reason_if_zero").reset_index(name="count")
    )
    print("[summary_counts_by_reason_if_zero]")
    print(reason_counts.to_string(index=False))

    cols = [
        "zotero_key",
        "text_path_exists",
        "total_n_csv_found",
        "reason_if_zero",
        "key2txt_used",
        "resolved_text_path",
        "candidate_dir_csv_counts",
        "sample_table_filenames",
    ]
    a = out_df[(out_df["text_path_exists"] == True) & (out_df["total_n_csv_found"].astype(int) == 0)].head(1)
    b = out_df[out_df["reason_if_zero"].astype(str) == "token_match_found_elsewhere_only"].head(1)
    c = out_df[out_df["total_n_csv_found"].astype(int) > 0].head(1)
    print("[surprising_a_text_exists_but_zero]")
    print(a[cols].to_string(index=False) if not a.empty else "(empty)")
    print("[surprising_b_token_match_found_elsewhere_only]")
    print(b[cols].to_string(index=False) if not b.empty else "(empty)")
    print("[surprising_c_has_csvs_found]")
    print(c[cols].to_string(index=False) if not c.empty else "(empty)")

    if args.dry_run:
        print("[dry_run_scanned_dirs_and_first10_csv]")
        for _, r in out_df.head(min(3, len(out_df))).iterrows():
            print(f"key={r.get('zotero_key','')}")
            print(f"scanned_dir={r.get('candidate_tables_dirs','')}")
            print(f"first10_csv={r.get('sample_table_filenames','')}")


if __name__ == "__main__":
    main()
