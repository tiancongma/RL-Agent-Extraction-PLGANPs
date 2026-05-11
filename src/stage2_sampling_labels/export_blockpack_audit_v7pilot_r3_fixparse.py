#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from src.stage2_sampling_labels.auto_extract_weak_labels_v7pilot_r3_fixparse import (
    EvidenceBlock,
    build_evidence_candidates,
    count_source_line,
    format_evidence_block,
    pack_evidence_blocks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export deterministic block-packing audit artifacts for one paper.")
    p.add_argument("--paper-key", default="L3H2RS2H")
    p.add_argument("--doi", default="10.1016/j.ejpb.2004.09.002")
    p.add_argument(
        "--title",
        default="Development and characterization of PLGA nanospheres and nanocapsules containing xanthone and 3-methoxyxanthone",
    )
    p.add_argument(
        "--text-path",
        default="data/cleaned/content_goren_2025/text/L3H2RS2H.pdf.txt",
    )
    p.add_argument(
        "--out-dir",
        default="data/cleaned/labels/manual/l3h2rs2h_blockpack_audit_2026-03-10_v2",
    )
    p.add_argument(
        "--note-md",
        default="docs/methods/l3h2rs2h_blockpack_audit_2026-03-10_v2.md",
    )
    p.add_argument(
        "--previous-inventory-tsv",
        default="data/cleaned/labels/manual/l3h2rs2h_blockpack_audit_2026-03-10/block_inventory.tsv",
    )
    p.add_argument(
        "--previous-order-tsv",
        default="data/cleaned/labels/manual/l3h2rs2h_blockpack_audit_2026-03-10/packed_block_order.tsv",
    )
    p.add_argument("--max-chars", type=int, default=50000)
    return p.parse_args()


def preview_text(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def block_label(block_type: str) -> str:
    return {
        "metadata": "METADATA",
        "synthesis_method": "SYNTHESIS_METHOD_BLOCK",
        "table": "TABLE_BLOCK",
        "caption": "CAPTION_BLOCK",
        "paragraph": "PARAGRAPH_BLOCK",
    }.get(block_type, block_type.upper())


def block_quality_label(block: EvidenceBlock) -> str:
    lower = block.text.lower()
    if block.block_type != "table":
        return "context"
    if "download:" in lower or "figure " in lower:
        return "noisy"
    if "m. teixeira et al." in lower or "journal of pharmaceutics" in lower:
        return "noisy"
    if len(block.text) > 1800:
        return "noisy"
    if lower.count("theoretical concentration") >= 2 or ("diameter" in lower and "zeta" in lower):
        return "likely_true_formulation_table"
    if "xan" in lower or "3-meoxan" in lower or "empty nanocapsules" in lower:
        return "likely_true_formulation_table"
    return "unclear_table"


def suspicious_reason(block: EvidenceBlock) -> str:
    lower = block.text.lower()
    reasons: List[str] = []
    if block.block_type == "table":
        if "download:" in lower:
            reasons.append("contains figure/download artifact")
        if "figure " in lower:
            reasons.append("caption-body mixture with figure text")
        if "m. teixeira et al." in lower:
            reasons.append("contains running paper header/footer text")
        if len(block.text) > 1800:
            reasons.append("table block is unusually long and likely includes narrative spillover")
        if lower.count(". ") >= 6:
            reasons.append("contains many narrative sentences for a table block")
    if block.block_type == "paragraph":
        if "table " not in lower and "xan" not in lower and "3-meoxan" not in lower and "empty" not in lower:
            reasons.append("generic paragraph with weak formulation linkage")
    return "; ".join(reasons)


def write_tsv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_tsv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def main() -> None:
    args = parse_args()
    text_path = Path(args.text_path)
    out_dir = Path(args.out_dir)
    note_md = Path(args.note_md)
    previous_inventory_tsv = Path(args.previous_inventory_tsv)
    previous_order_tsv = Path(args.previous_order_tsv)
    out_dir.mkdir(parents=True, exist_ok=True)
    note_md.parent.mkdir(parents=True, exist_ok=True)

    raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
    normalized_text, candidates = build_evidence_candidates(raw_text, args.paper_key, args.doi, args.title)
    packed = pack_evidence_blocks(candidates, max_chars=args.max_chars)
    selected = {
        item["block"].block_id: item
        for item in packed["selected_blocks"]
    }

    inventory_rows: List[Dict[str, Any]] = []
    packed_rows: List[Dict[str, Any]] = []

    for block in candidates:
        selected_item = selected.get(block.block_id)
        source_line = count_source_line(normalized_text, block.source_start)
        inventory_rows.append(
            {
                "paper_key": args.paper_key,
                "block_id": block.block_id,
                "block_type": block.block_type,
                "score": block.score,
                "char_len": len(block.text),
                "selected_for_packing": "yes" if selected_item else "no",
                "packing_rank": selected_item["packing_rank"] if selected_item else "",
                "source_start_line": source_line if source_line > 0 else "",
                "source_start_char": block.source_start if block.source_start >= 0 else "",
                "preview_text": preview_text(block.text),
                "quality_label": block_quality_label(block),
                "suspicious_reason": suspicious_reason(block),
            }
        )

    for item in packed["selected_blocks"]:
        block = item["block"]
        packed_rows.append(
            {
                "packing_rank": item["packing_rank"],
                "block_id": block.block_id,
                "block_type": block.block_type,
                "score": block.score,
                "char_len": item["char_len"],
                "cumulative_char_count": item["cumulative_char_count"],
                "truncated": item["truncated"],
                "preview_text": preview_text(block.text),
            }
        )

    inventory_path = out_dir / "block_inventory.tsv"
    order_path = out_dir / "packed_block_order.tsv"
    packed_text_path = out_dir / "packed_evidence_text.txt"

    write_tsv(
        inventory_path,
        inventory_rows,
        [
            "paper_key",
            "block_id",
            "block_type",
            "score",
            "char_len",
            "selected_for_packing",
            "packing_rank",
            "source_start_line",
            "source_start_char",
            "preview_text",
            "quality_label",
            "suspicious_reason",
        ],
    )
    write_tsv(
        order_path,
        packed_rows,
        [
            "packing_rank",
            "block_id",
            "block_type",
            "score",
            "char_len",
            "cumulative_char_count",
            "truncated",
            "preview_text",
        ],
    )

    selected_blocks = [item["block"] for item in packed["selected_blocks"]]
    previous_inventory = {row.get("block_id", ""): row for row in load_tsv_rows(previous_inventory_tsv)}
    previous_order = {row.get("block_id", ""): row for row in load_tsv_rows(previous_order_tsv)}
    summary_lines = [
        "[PACKING_SUMMARY]",
        f"paper_key: {args.paper_key}",
        f"doi: {args.doi}",
        f"text_path: {text_path}",
        f"candidate_block_count: {len(candidates)}",
        f"selected_block_count: {len(selected_blocks)}",
        f"packed_char_len: {len(str(packed['packed_text']))}",
        "",
    ]
    for item in packed["selected_blocks"]:
        block = item["block"]
        summary_lines.append(f"[BLOCK {item['packing_rank']}][{block_label(block.block_type)}]")
        summary_lines.append(
            f"block_id={block.block_id} score={block.score} char_len={item['char_len']} "
            f"source_start_line={count_source_line(normalized_text, block.source_start) if block.source_start >= 0 else ''} "
            f"truncated={item['truncated']}"
        )
        summary_lines.append(block.text)
        summary_lines.append("")
    packed_text_path.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")

    selected_paragraph_ranks = [
        item["packing_rank"] for item in packed["selected_blocks"] if item["block"].block_type in {"paragraph", "synthesis_method"}
    ]
    synthesis_rows = [
        row for row in inventory_rows if row["selected_for_packing"] == "yes" and row["block_type"] == "synthesis_method"
    ]
    suspicious_blocks = [
        row for row in inventory_rows if row["selected_for_packing"] == "yes" and row["suspicious_reason"]
    ]
    suspicious_blocks = sorted(suspicious_blocks, key=lambda r: (-(int(r["score"]) if str(r["score"]).strip() else 0), str(r["block_id"])))
    suspicious_blocks = suspicious_blocks[:8]

    lines: List[str] = []
    lines.append("# L3H2RS2H Block Packing Audit (2026-03-10)")
    lines.append("")
    lines.append("## Overall readout")
    selected_tables = [row for row in inventory_rows if row["selected_for_packing"] == "yes" and row["block_type"] == "table"]
    likely_true = [row for row in selected_tables if row["quality_label"] == "likely_true_formulation_table"]
    noisy = [row for row in selected_tables if row["quality_label"] == "noisy"]
    lines.append(
        f"- Selected table blocks: {len(selected_tables)} total; likely true formulation tables: {len(likely_true)}; likely noisy table blocks: {len(noisy)}."
    )
    lines.append(f"- Selected synthesis-method blocks: {len(synthesis_rows)}.")
    if selected_paragraph_ranks:
        lines.append(f"- First selected paragraph block appears at packing rank {min(selected_paragraph_ranks)}.")
    else:
        lines.append("- No paragraph blocks were selected.")
    if synthesis_rows:
        promoted_ids = ", ".join(row["block_id"] for row in synthesis_rows[:6])
        lines.append(f"- Promoted synthesis-method blocks: {promoted_ids}.")
    lines.append("- Best current hypothesis: synthesis-defining preparation paragraphs should now anchor family/grouping logic before enumeration-heavy tables.")
    lines.append("")
    lines.append("## Suspicious selected blocks")
    if suspicious_blocks:
        for row in suspicious_blocks:
            lines.append(
                f"- `{row['block_id']}` ({row['block_type']}, score={row['score']}): {row['suspicious_reason']}. Preview: {row['preview_text']}"
            )
    else:
        lines.append("- No selected blocks were flagged as suspicious by the simple audit heuristic.")
    lines.append("")
    lines.append("## Before vs after suspicious blocks")
    for block_id in ["table_5", "table_9", "table_8", "table_6"]:
        current = next((row for row in inventory_rows if row["block_id"] == block_id), None)
        previous = previous_inventory.get(block_id)
        if not current and not previous:
            lines.append(f"- `{block_id}`: not present in either audit export.")
            continue
        if previous and not current:
            lines.append(f"- `{block_id}`: rejected in the cleaned pass; previously selected with char_len={previous.get('char_len', '')} and quality={previous.get('quality_label', '')}.")
            continue
        if current and not previous:
            lines.append(f"- `{block_id}`: new block in cleaned pass with char_len={current['char_len']} and quality={current['quality_label']}.")
            continue
        prev_len = int(previous.get("char_len", "0") or 0)
        cur_len = int(current.get("char_len", "0") or 0)
        prev_rank = previous_order.get(block_id, {}).get("packing_rank", "")
        cur_rank = current.get("packing_rank", "")
        changes: List[str] = []
        if cur_len < prev_len:
            changes.append(f"shorter ({prev_len} -> {cur_len})")
        elif cur_len > prev_len:
            changes.append(f"longer ({prev_len} -> {cur_len})")
        if current["quality_label"] != previous.get("quality_label", ""):
            changes.append(f"quality {previous.get('quality_label', '')} -> {current['quality_label']}")
        if current["selected_for_packing"] != previous.get("selected_for_packing", ""):
            changes.append(f"selected {previous.get('selected_for_packing', '')} -> {current['selected_for_packing']}")
        if cur_rank != prev_rank:
            changes.append(f"packing_rank {prev_rank or 'blank'} -> {cur_rank or 'blank'}")
        if not changes:
            changes.append("still broadly similar")
        lines.append(f"- `{block_id}`: " + "; ".join(changes) + ".")
    lines.append("")
    lines.append("## Audit answers")
    synth_before_tables = [row for row in synthesis_rows if row.get("packing_rank") and int(row["packing_rank"]) < min([int(t["packing_rank"]) for t in selected_tables] or [9999])]
    lines.append(
        f"- Did one or more `SYNTHESIS_METHOD_BLOCK`s appear before the table blocks? {'yes' if synth_before_tables else 'no'}."
    )
    lines.append(
        f"- Which paragraphs were promoted? {', '.join(row['block_id'] for row in synthesis_rows) if synthesis_rows else 'none'}."
    )
    lines.append(
        f"- True formulation-table candidates among selected table blocks: {', '.join(r['block_id'] for r in likely_true) if likely_true else 'none clearly isolated'}."
    )
    lines.append(
        f"- Noisy/fragmentary selected table blocks: {', '.join(r['block_id'] for r in noisy) if noisy else 'none strongly flagged'}."
    )
    late_para = min(selected_paragraph_ranks) if selected_paragraph_ranks else ""
    previous_paragraph_rank = ""
    for row in load_tsv_rows(previous_order_tsv):
        if row.get("block_type") == "paragraph":
            previous_paragraph_rank = row.get("packing_rank", "")
            break
    if late_para and late_para > 4:
        if previous_paragraph_rank:
            lines.append(f"- Did the first explanatory/preparation paragraph move substantially earlier? {'yes' if int(previous_paragraph_rank) - int(late_para) >= 5 else 'no'}. Current rank {late_para}; previous rank {previous_paragraph_rank}.")
        else:
            lines.append(f"- Did the first explanatory/preparation paragraph move substantially earlier? current rank {late_para}.")
    else:
        if previous_paragraph_rank:
            lines.append(f"- Did the first explanatory/preparation paragraph move substantially earlier? {'yes' if late_para and int(previous_paragraph_rank) - int(late_para) >= 5 else 'no'}. Current rank {late_para or 'n/a'}; previous rank {previous_paragraph_rank}.")
        else:
            lines.append(f"- Did the first explanatory/preparation paragraph move substantially earlier? current rank {late_para or 'n/a'}.")
    lines.append("- Does the packed evidence now look more suitable for preserving parent/variant reasoning? yes, if the promoted synthesis-method blocks appear before the table stack and describe shared/fixed preparation logic.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Block inventory: `{inventory_path}`")
    lines.append(f"- Packed order: `{order_path}`")
    lines.append(f"- Packed evidence text: `{packed_text_path}`")
    note_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"block_inventory={inventory_path}")
    print(f"packed_block_order={order_path}")
    print(f"packed_evidence_text={packed_text_path}")
    print(f"note_md={note_md}")


if __name__ == "__main__":
    main()
