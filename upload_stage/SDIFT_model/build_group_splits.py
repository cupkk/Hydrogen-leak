import argparse
import json

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["case_id", "space_id", "obstacle_id", "vent_id"]


def check_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")


def split_train_val(case_ids, val_ratio=0.1, seed=231):
    case_ids = np.array(case_ids)
    if case_ids.size == 0:
        return [], []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(case_ids.size)
    n_val = int(np.floor(case_ids.size * val_ratio))
    if case_ids.size > 1:
        n_val = max(1, n_val)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.size == 0:
        train_idx = val_idx
        val_idx = np.array([], dtype=np.int64)
    return case_ids[train_idx].tolist(), case_ids[val_idx].tolist()


def make_axis_splits(df, axis, val_ratio=0.1, seed=231):
    groups = sorted(df[axis].astype(str).unique().tolist())
    out = []
    for i, g in enumerate(groups):
        is_test = df[axis].astype(str) == g
        test_case_ids = df.loc[is_test, "case_id"].tolist()
        train_val_case_ids = df.loc[~is_test, "case_id"].tolist()
        train_case_ids, val_case_ids = split_train_val(train_val_case_ids, val_ratio=val_ratio, seed=seed + i)
        out.append(
            {
                "axis": axis,
                "held_out_group": g,
                "train_case_ids": train_case_ids,
                "val_case_ids": val_case_ids,
                "test_case_ids": test_case_ids,
            }
        )
    return out


def make_learning_curve_subsets(train_case_ids, sizes, seed=231):
    rng = np.random.default_rng(seed)
    train_case_ids = np.array(sorted(set(train_case_ids)))
    perm = rng.permutation(train_case_ids.size)
    shuffled = train_case_ids[perm]
    out = []
    for n in sizes:
        n_eff = int(min(max(n, 1), shuffled.size))
        out.append({"size": int(n_eff), "case_ids": shuffled[:n_eff].tolist()})
    return out


def main():
    parser = argparse.ArgumentParser(description="Build scene/group splits for generalization evaluation.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--learning_curve_sizes", type=int, nargs="+", default=[50, 100, 200, 300, 500, 800])
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    check_columns(df)
    df["case_id"] = df["case_id"].astype(str)

    by_axis = {}
    for axis in ["space_id", "obstacle_id", "vent_id"]:
        by_axis[axis] = make_axis_splits(df, axis, val_ratio=args.val_ratio, seed=args.seed)

    all_case_ids = df["case_id"].tolist()
    train_case_ids, val_case_ids = split_train_val(all_case_ids, val_ratio=args.val_ratio, seed=args.seed)
    baseline = {
        "train_case_ids": train_case_ids,
        "val_case_ids": val_case_ids,
        "test_case_ids": [],
    }
    learning_curve = make_learning_curve_subsets(train_case_ids, args.learning_curve_sizes, seed=args.seed)

    payload = {
        "manifest_csv": args.manifest_csv,
        "required_columns": REQUIRED_COLUMNS,
        "baseline_iid": baseline,
        "group_splits": by_axis,
        "learning_curve_subsets": learning_curve,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.out_csv:
        rows = []
        for axis, items in by_axis.items():
            for item in items:
                rows.append(
                    {
                        "axis": axis,
                        "held_out_group": item["held_out_group"],
                        "train_count": len(item["train_case_ids"]),
                        "val_count": len(item["val_case_ids"]),
                        "test_count": len(item["test_case_ids"]),
                    }
                )
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f"saved: {args.out_json}")
    if args.out_csv:
        print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
