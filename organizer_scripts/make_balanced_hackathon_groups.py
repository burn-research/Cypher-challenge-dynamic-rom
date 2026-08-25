#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create balanced hackathon groups from an Excel file.

The script reads participants and their coding level, then assigns participants
into groups with balanced total skill while keeping group sizes close to 4.

Example:
    python organizer_scripts/make_balanced_hackathon_groups.py \
        --input participants.xlsx \
        --name-column fullName \
        --skill-column Coding Level \
        --output groups.xlsx
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_SKILL_MAP: Dict[str, float] = {
    "No experience": 0.0,
    "Beginner (basic syntax, variables, loops)": 1.0,
    "Intermediate (functions, libraries like NumPy/Pandas, debug)": 2.0,
    "Advanced (scientific computing, ML workflows, optimization, packages)": 3.0,
}

NORMALIZED_SKILL_MAP: Dict[str, float] = {
    str(k).strip().lower(): float(v) for k, v in DEFAULT_SKILL_MAP.items()
}


@dataclass
class GroupState:
    members: List[int]
    skill_sum: float
    capacity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build balanced hackathon groups from an Excel participant list."
    )
    parser.add_argument("--input", required=True, help="Path to input Excel file.")
    parser.add_argument(
        "--output",
        default="balanced_groups.xlsx",
        help="Path to output Excel file (default: balanced_groups.xlsx).",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index in the Excel file (default: first sheet).",
    )
    parser.add_argument(
        "--name-column",
        default="name",
        help="Column containing participant names (default: name).",
    )
    parser.add_argument(
        "--skill-column",
        default="coding_level",
        help="Column containing coding skill (default: coding_level).",
    )
    parser.add_argument(
        "--target-group-size",
        type=int,
        default=4,
        help="Preferred group size (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking among equal skill values (default: 42).",
    )
    return parser.parse_args()


def _to_skill_score(value: object, skill_map: Dict[str, float]) -> float:
    if pd.isna(value):
        raise ValueError("Found empty skill value in the input data.")

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()
    try:
        return float(value_str)
    except ValueError:
        pass

    mapped = skill_map.get(value_str.lower())
    if mapped is not None:
        return mapped

    supported = ", ".join(sorted(skill_map.keys()))
    raise ValueError(
        f"Unsupported skill value '{value_str}'. "
        f"Use numeric values or one of: {supported}."
    )


def normalize_skills(df: pd.DataFrame, skill_column: str) -> pd.Series:
    return df[skill_column].apply(lambda val: _to_skill_score(val, NORMALIZED_SKILL_MAP))


def compute_group_capacities(n_people: int, target_size: int) -> List[int]:
    if target_size <= 0:
        raise ValueError("target_group_size must be a positive integer.")

    n_groups = math.ceil(n_people / target_size)
    base_size = n_people // n_groups
    remainder = n_people % n_groups

    capacities = [base_size + 1 if idx < remainder else base_size for idx in range(n_groups)]
    return capacities


def assign_balanced_groups(df: pd.DataFrame, target_group_size: int) -> pd.DataFrame:
    n_people = len(df)
    capacities = compute_group_capacities(n_people, target_group_size)

    groups: List[GroupState] = [
        GroupState(members=[], skill_sum=0.0, capacity=cap) for cap in capacities
    ]

    sorted_df = df.sort_values(
        by=["skill_score", "_shuffle"], ascending=[False, True], ignore_index=False
    )

    for row_idx, row in sorted_df.iterrows():
        # Pick the group with the lowest current total skill among groups with room.
        candidate_indices = [
            i for i, g in enumerate(groups) if len(g.members) < g.capacity
        ]
        selected_group = min(
            candidate_indices,
            key=lambda i: (groups[i].skill_sum, len(groups[i].members), i),
        )

        groups[selected_group].members.append(row_idx)
        groups[selected_group].skill_sum += float(row["skill_score"])

    out = df.copy()
    out["group"] = -1
    for g_idx, group in enumerate(groups, start=1):
        out.loc[group.members, "group"] = g_idx

    return out.sort_values(["group", "skill_score"], ascending=[True, False])


def build_hackathon_table(assigned: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        assigned.sort_values(["group", "skill_score"], ascending=[True, False])
        .groupby("group")["participant"]
        .apply(list)
    )

    max_members = max((len(members) for members in grouped), default=0)
    # Keep the requested layout: first column is group number and columns 2..6 are members.
    max_members = max(max_members, 5)

    rows = []
    for group_num, members in grouped.items():
        row = {"Group number": int(group_num)}
        for i in range(max_members):
            row[f"Member {i + 1}"] = members[i] if i < len(members) else ""
        rows.append(row)

    columns = ["Group number"] + [f"Member {i + 1}" for i in range(max_members)]
    return pd.DataFrame(rows, columns=columns)


def validate_columns(df: pd.DataFrame, name_column: str, skill_column: str) -> None:
    missing = [col for col in [name_column, skill_column] if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            f"{', '.join(missing)}. Available columns: {', '.join(df.columns)}"
        )


def main() -> None:
    args = parse_args()

    if args.target_group_size < 2:
        raise ValueError("target_group_size should be at least 2.")

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    validate_columns(df, args.name_column, args.skill_column)

    # Ignore incomplete rows (common in exported registration spreadsheets).
    before_rows = len(df)
    df = df.dropna(subset=[args.name_column, args.skill_column]).copy()
    after_rows = len(df)
    dropped_rows = before_rows - after_rows
    if dropped_rows > 0:
        print(
            f"Skipped {dropped_rows} row(s) with missing "
            f"{args.name_column} or {args.skill_column}."
        )

    if after_rows == 0:
        raise ValueError(
            "No valid participants left after dropping rows with missing "
            "name or skill."
        )

    work = df[[args.name_column, args.skill_column]].copy()
    work = work.rename(
        columns={args.name_column: "participant", args.skill_column: "skill_raw"}
    )
    work["skill_score"] = normalize_skills(work, "skill_raw")
    work["_shuffle"] = work.sample(frac=1.0, random_state=args.seed).index

    assigned = assign_balanced_groups(work, target_group_size=args.target_group_size)

    summary = (
        assigned.groupby("group", as_index=False)
        .agg(
            group_size=("participant", "count"),
            total_skill=("skill_score", "sum"),
            avg_skill=("skill_score", "mean"),
        )
        .sort_values("group")
    )
    hackathon_table = build_hackathon_table(assigned)

    output_cols = ["group", "participant", "skill_raw", "skill_score"]
    with pd.ExcelWriter(args.output) as writer:
        assigned[output_cols].to_excel(writer, index=False, sheet_name="assignments")
        summary.to_excel(writer, index=False, sheet_name="summary")

    hackathon_output = str(Path(args.output).with_name("Groups_hackathon.xlsx"))
    hackathon_table.to_excel(hackathon_output, index=False)

    print(f"Created {len(summary)} groups and saved results to: {args.output}")
    print(f"Saved hackathon table to: {hackathon_output}")
    print("Group sizes:", ", ".join(str(x) for x in summary["group_size"].tolist()))


if __name__ == "__main__":
    main()
