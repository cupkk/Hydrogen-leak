from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "report_figures_20260413"
SERVER_RESULTS_DIR = DOCS_DIR / "server_results_20260413"
SDIFT_ROOT = sorted(ROOT.glob("SDIFT*"))[0]
DATA_DIR = SDIFT_ROOT / "data"


def configure_matplotlib() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.sans-serif"] = preferred_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 160
    matplotlib.rcParams["savefig.dpi"] = 200


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def make_validation_loop_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = "导师要求的可验证闭环"
    subtitle = "不是只看花哨图，而是要在可控真值下证明：模型从稀疏传感器输入出发，能回到 CFD 全场并给出定量误差。"
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.89, subtitle, ha="center", va="top", fontsize=12, color="#4b5563")

    box_specs = [
        (0.03, 0.56, 0.18, 0.18, "#dbeafe", "1. CFD 全场真值", "三维浓度场\n作为可控“标准答案”"),
        (0.25, 0.56, 0.18, 0.18, "#dcfce7", "2. 抽样成传感器序列", "从全场提取若干测点\n形成浓度-时间输入"),
        (0.47, 0.56, 0.18, 0.18, "#fef3c7", "3. 输入 SDIFT 模型", "FTM + GPSD + MPDPS\n执行三维场反演"),
        (0.69, 0.56, 0.18, 0.18, "#ede9fe", "4. 得到反演浓度场", "输出 120 s × 48³\n时空浓度场"),
    ]

    for x, y, w, h, color, header, body in box_specs:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.68, header, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center", fontsize=11, color="#1f2937")

    for x0, x1 in [(0.21, 0.25), (0.43, 0.47), (0.65, 0.69)]:
        ax.annotate("", xy=(x1, 0.65), xytext=(x0, 0.65), arrowprops=dict(arrowstyle="->", lw=2.0, color="#475569"))

    final_rect = patches.FancyBboxPatch(
        (0.28, 0.24),
        0.44,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor="#7c2d12",
        facecolor="#ffedd5",
    )
    ax.add_patch(final_rect)
    ax.text(0.5, 0.34, "5. 回到 CFD 全场逐点对比", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.28, "用统一指标衡量模型是否真的反演正确，而不是靠肉眼判断“像不像”。", ha="center", va="center", fontsize=11)

    metric_specs = [
        (0.07, 0.06, "RMSE", "整体误差幅度"),
        (0.29, 0.06, "MAE", "平均绝对偏差"),
        (0.51, 0.06, "有效羽流区相对 L1", "更关注真正有浓度的区域"),
        (0.73, 0.06, "质量积分相对误差", "检验全场尺度是否偏大/偏小"),
    ]
    for x, y, header, body in metric_specs:
        rect = patches.FancyBboxPatch(
            (x, y),
            0.18,
            0.1,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#94a3b8",
            facecolor="#f8fafc",
        )
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.065, header, ha="center", va="center", fontsize=11.5, fontweight="bold")
        ax.text(x + 0.09, y + 0.028, body, ha="center", va="center", fontsize=9.5, color="#475569")

    save_fig(fig, "teacher_validation_loop.png")


def make_clean48_coverage_split_figure() -> None:
    manifest = pd.read_csv(DATA_DIR / "cfd48_clean_T120_interp48_manifest.csv")
    pos_train = pd.read_csv(DATA_DIR / "splits_clean" / "holdout_400_0_0_val_300_0_0" / "train_manifest.csv")
    pos_val = pd.read_csv(DATA_DIR / "splits_clean" / "holdout_400_0_0_val_300_0_0" / "val_manifest.csv")
    pos_test = pd.read_csv(DATA_DIR / "splits_clean" / "holdout_400_0_0_val_300_0_0" / "test_manifest.csv")

    grouped = (
        manifest.groupby(["source_x_mm", "source_y_mm"], as_index=False)
        .agg(case_count=("case_id", "count"))
        .sort_values(["source_x_mm", "source_y_mm"])
    )
    manifest["position_label"] = manifest.apply(
        lambda r: f"({int(r['source_x_mm'])},{int(r['source_y_mm'])},0)", axis=1
    )
    rates = sorted(manifest["leak_rate_ml_min"].astype(int).unique())
    position_order = (
        manifest[["source_x_mm", "source_y_mm", "position_label"]]
        .drop_duplicates()
        .sort_values(["source_x_mm", "source_y_mm"])
    )
    pivot = (
        manifest.assign(present=1)
        .pivot_table(index="position_label", columns="leak_rate_ml_min", values="present", aggfunc="max", fill_value=0)
        .reindex(index=position_order["position_label"].tolist(), columns=rates)
    )

    train_pos = set(pos_train["position_key"].astype(str))
    val_pos = set(pos_val["position_key"].astype(str))
    test_pos = set(pos_test["position_key"].astype(str))

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.set_title("Clean48 数据位置覆盖与 holdout 划分", fontsize=15, fontweight="bold")
    for _, row in grouped.iterrows():
        x = row["source_x_mm"]
        y = row["source_y_mm"]
        key = f"{int(x)},{int(y)},0"
        if key in test_pos:
            color = "#dc2626"
            marker = "*"
            size = 420
            label = "test 位置 (400,0,0)"
        elif key in val_pos:
            color = "#f59e0b"
            marker = "s"
            size = 240
            label = "val 位置 (300,0,0)"
        else:
            color = "#2563eb"
            marker = "o"
            size = 120 + row["case_count"] * 30
            label = "train 位置"
        ax0.scatter(x, y, s=size, c=color, marker=marker, edgecolors="black", linewidths=0.8, zorder=3, label=label)
        ax0.text(x + 8, y + 8, f"({int(x)},{int(y)})\n{int(row['case_count'])}组", fontsize=9)

    handles, labels = ax0.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax0.legend(dedup.values(), dedup.keys(), loc="lower left", fontsize=10, frameon=True)
    ax0.set_xlabel("源位置 x / mm")
    ax0.set_ylabel("源位置 y / mm")
    ax0.grid(alpha=0.25, linestyle="--")
    ax0.set_xlim(-30, 430)
    ax0.set_ylim(-340, 40)

    im = ax1.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax1.set_title("Clean48 位置-泄漏率覆盖矩阵", fontsize=15, fontweight="bold")
    ax1.set_xticks(np.arange(len(rates)))
    ax1.set_xticklabels(rates)
    ax1.set_yticks(np.arange(len(pivot.index)))
    ax1.set_yticklabels(pivot.index)
    ax1.set_xlabel("泄漏率 / mL/min")
    ax1.set_ylabel("泄漏位置")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iloc[i, j])
            ax1.text(j, i, "Y" if val else "", ha="center", va="center", color="#111827", fontsize=10, fontweight="bold")

    test_row_idx = list(pivot.index).index("(400,0,0)")
    ax1.add_patch(patches.Rectangle((-0.5, test_row_idx - 0.5), len(rates), 1, fill=False, edgecolor="#dc2626", linewidth=2.4))
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="是否存在该工况")

    fig.suptitle("Clean48 数据覆盖与正式位置 holdout 划分", fontsize=18, fontweight="bold", y=1.02)
    save_fig(fig, "clean48_coverage_split.png")


def _sort_position_label(x_mm: int, y_mm: int) -> tuple[int, int]:
    return int(x_mm), int(y_mm)


def make_clean48_formal_casewise_figure() -> None:
    pos_test = load_json(
        SERVER_RESULTS_DIR / "results__advisor_study__clean48_holdout400_formal_20260413_fix__test_eval__aggregate_metrics.json"
    )
    rate_test = load_json(
        SERVER_RESULTS_DIR / "results__advisor_study__clean48_holdout_rate0100_formal_20260413_fix__test_eval__aggregate_metrics.json"
    )

    pos_rows = sorted(pos_test["rows"], key=lambda r: int(r["leak_rate_ml_min"]))
    rate_rows = sorted(rate_test["rows"], key=lambda r: _sort_position_label(r["source_x_mm"], r["source_y_mm"]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: unseen position
    ax = axes[0]
    x = np.arange(len(pos_rows))
    rmse = [r["global_rmse"] for r in pos_rows]
    mae = [r["global_mae"] for r in pos_rows]
    active_l1 = [r["global_rel_l1_active_mean"] for r in pos_rows]
    labels = [str(int(r["leak_rate_ml_min"])) for r in pos_rows]
    width = 0.38
    ax.bar(x - width / 2, rmse, width=width, color="#2563eb", label="RMSE")
    ax.bar(x + width / 2, mae, width=width, color="#60a5fa", label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("未见位置 (400,0,0) 的泄漏率 / mL/min")
    ax.set_ylabel("绝对误差")
    ax.set_title("未见位置 test：不同泄漏率下的重建误差", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25, axis="y", linestyle="--")
    ax2 = ax.twinx()
    ax2.plot(x, active_l1, color="#dc2626", marker="o", linewidth=2.0, label="有效羽流区相对 L1")
    ax2.set_ylabel("有效羽流区相对 L1")
    ax.annotate(
        "低泄漏率绝对误差仍低，但相对误差更高",
        xy=(0, rmse[0]),
        xytext=(0.6, max(rmse) * 0.85),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#b91c1c"),
        fontsize=10,
        color="#b91c1c",
    )
    lines, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # Right panel: unseen leak rate
    ax = axes[1]
    x = np.arange(len(rate_rows))
    rmse = [r["global_rmse"] for r in rate_rows]
    mae = [r["global_mae"] for r in rate_rows]
    labels = [f"({int(r['source_x_mm'])},{int(r['source_y_mm'])})" for r in rate_rows]
    ax.bar(x - width / 2, rmse, width=width, color="#059669", label="RMSE")
    ax.bar(x + width / 2, mae, width=width, color="#6ee7b7", label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel("未见泄漏率 100 mL/min 的测试位置")
    ax.set_ylabel("绝对误差")
    ax.set_title("未见泄漏率 test：不同位置下的重建误差", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25, axis="y", linestyle="--")
    max_idx = int(np.argmax(rmse))
    ax.annotate(
        "该位置最难，提示空间位置差异\n仍会影响固定泄漏率下的反演难度",
        xy=(max_idx, rmse[max_idx]),
        xytext=(max_idx - 2.0, max(rmse) * 0.78),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#065f46"),
        fontsize=10,
        color="#065f46",
    )
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Clean48 正式闭环结果：case 级直观展示", fontsize=18, fontweight="bold", y=1.02)
    save_fig(fig, "clean48_formal_casewise.png")


def make_teacher_question_response_figure() -> None:
    pos_test = load_json(
        SERVER_RESULTS_DIR / "results__advisor_study__clean48_holdout400_formal_20260413_fix__test_eval__aggregate_metrics.json"
    )
    rate_test = load_json(
        SERVER_RESULTS_DIR / "results__advisor_study__clean48_holdout_rate0100_formal_20260413_fix__test_eval__aggregate_metrics.json"
    )
    train_size = pd.read_csv(SERVER_RESULTS_DIR / "clean48_train_scale_final_summary.csv")

    best_row = train_size.loc[train_size["rmse_mean"].idxmin()]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "对导师三类问题的当前阶段性回答", ha="center", va="top", fontsize=22, fontweight="bold")

    cards = [
        (
            0.04,
            0.18,
            0.27,
            0.62,
            "#ecfeff",
            "问题 1：模型是否真的起作用？",
            [
                "已回答到“同尺度 clean 数据下有效”。",
                f"未见位置 test RMSE = {pos_test['metrics']['global_rmse']['mean']:.3e}",
                f"未见泄漏率 test RMSE = {rate_test['metrics']['global_rmse']['mean']:.3e}",
                "说明模型不仅能出图，还能在未见工况上回到 CFD 真值做定量对比。",
            ],
        ),
        (
            0.365,
            0.18,
            0.27,
            0.62,
            "#f0fdf4",
            "问题 2：模型好坏如何量化？",
            [
                "已建立统一评价体系。",
                "RMSE / MAE：衡量整体绝对误差。",
                "有效羽流区相对 L1：更关注真正有浓度的位置。",
                "质量积分相对误差：判断全场尺度是否偏大或偏小。",
            ],
        ),
        (
            0.69,
            0.18,
            0.27,
            0.62,
            "#fff7ed",
            "问题 3：下一步还差什么？",
            [
                f"训练数据量实验当前最佳均值出现在 n={int(best_row['train_size'])}。",
                "但关系呈非单调波动，说明瓶颈已转向训练稳定性与观测注入。",
                "尺寸泛化仍未完成，需要新增不同箱体尺寸的 CFD 数据。",
                "真实实验数据更适合作为外部盲测，而不是立刻混入主训练集。",
            ],
        ),
    ]

    for x, y, w, h, color, header, bullets in cards:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.07, header, ha="center", va="top", fontsize=14, fontweight="bold")
        yy = y + h - 0.15
        for bullet in bullets:
            ax.text(x + 0.02, yy, f"• {bullet}", ha="left", va="top", fontsize=11, color="#1f2937")
            yy -= 0.11

    ax.text(
        0.5,
        0.07,
        "结论：现在已经足以做阶段性汇报，但必须把结论边界收紧到 clean 数据、同尺度箱体、未见位置/未见泄漏率闭环验证。",
        ha="center",
        va="center",
        fontsize=12,
        color="#374151",
    )
    save_fig(fig, "teacher_question_response.png")


def main() -> None:
    configure_matplotlib()
    ensure_dirs()
    print("Generating teacher_validation_loop.png ...")
    make_validation_loop_figure()
    print("Generating clean48_coverage_split.png ...")
    make_clean48_coverage_split_figure()
    print("Generating clean48_formal_casewise.png ...")
    make_clean48_formal_casewise_figure()
    print("Generating teacher_question_response.png ...")
    make_teacher_question_response_figure()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
