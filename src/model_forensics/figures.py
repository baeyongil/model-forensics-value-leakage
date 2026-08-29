"""The three preregistered main figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


COLORS = {
    "baseline": "#6B7280",
    "threshold_only": "#9CA3AF",
    "above_good": "#0072B2",
    "below_good": "#D55E00",
    "accuracy_commitment": "#009E73",
    "value_threshold_planning": "#CC79A7",
    "epistemic_control": "#E69F00",
    "j": "#0072B2",
    "r": "#D55E00",
}


def _finish(fig: Any, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    return path


def plot_first_vs_final_bias(summary: pd.DataFrame, destination: str | Path) -> Path:
    """Plot good-side rates for first and final estimates by task/condition.

    Required columns: ``task``, ``condition``, ``stage``, ``rate``, ``ci_low``,
    and ``ci_high``. ``stage`` must contain ``first`` and ``final``.
    """

    required = {"task", "condition", "stage", "rate", "ci_low", "ci_high"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"behavior summary missing columns: {sorted(missing)}")
    plt = _pyplot()
    tasks = list(dict.fromkeys(summary["task"].astype(str)))
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.4), squeeze=False)
    for axis, task in zip(axes[0], tasks, strict=True):
        subset = summary[summary["task"] == task].copy()
        conditions = [
            condition
            for condition in ["baseline", "threshold_only", "above_good", "below_good"]
            if condition in set(subset["condition"])
        ]
        x = np.arange(len(conditions), dtype=float)
        for offset, stage, marker in [(-0.12, "first", "o"), (0.12, "final", "s")]:
            stage_rows = subset[subset["stage"] == stage].set_index("condition")
            values = np.array([stage_rows.loc[c, "rate"] for c in conditions], dtype=float)
            lows = np.array([stage_rows.loc[c, "ci_low"] for c in conditions], dtype=float)
            highs = np.array([stage_rows.loc[c, "ci_high"] for c in conditions], dtype=float)
            axis.errorbar(
                x + offset,
                values,
                yerr=np.vstack([values - lows, highs - values]),
                fmt=marker,
                markersize=6,
                capsize=3,
                color="#111827" if stage == "first" else "#0072B2",
                label=stage.capitalize(),
            )
        axis.axhline(0.5, color="#D1D5DB", linewidth=1, linestyle="--")
        axis.set_xticks(x, [c.replace("_", "\n") for c in conditions])
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("Probability estimate exceeds the threshold")
        axis.set_title(task.replace("_", " ").title())
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "When does the value-direction gap enter the estimate?", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    return _finish(fig, destination)


def plot_sentence_effect_forest(effects: pd.DataFrame, destination: str | Path) -> Path:
    """Forest plot of retain-minus-resample effects.

    Required columns: ``sentence_class``, ``direction``, ``estimate``,
    ``ci_low``, ``ci_high``, and ``conclusion``.
    """

    required = {"sentence_class", "direction", "estimate", "ci_low", "ci_high", "conclusion"}
    missing = required - set(effects.columns)
    if missing:
        raise ValueError(f"effect table missing columns: {sorted(missing)}")
    order = ["accuracy_commitment", "value_threshold_planning", "epistemic_control"]
    direction_order = ["above_good", "below_good", "pooled"]
    rows = effects.copy()
    rows["sentence_class"] = pd.Categorical(rows["sentence_class"], order, ordered=True)
    rows["direction"] = pd.Categorical(rows["direction"], direction_order, ordered=True)
    rows = rows.sort_values(["sentence_class", "direction"]).reset_index(drop=True)

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(8.2, max(4.3, 0.55 * len(rows) + 1.5)))
    positions = np.arange(len(rows))[::-1]
    for y, (_, row) in zip(positions, rows.iterrows(), strict=True):
        color = COLORS.get(str(row["sentence_class"]), "#4B5563")
        estimate = float(row["estimate"])
        axis.errorbar(
            estimate,
            y,
            xerr=[[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]],
            fmt="o",
            color=color,
            capsize=3,
            markersize=6,
        )
    axis.axvspan(-0.10, 0.10, color="#F3F4F6", zorder=-2, label="±10 pp ROPE")
    axis.axvline(0, color="#6B7280", linewidth=1)
    labels = [
        f"{str(row.sentence_class).replace('_', ' ')} · {str(row.direction).replace('_', ' ')}"
        for row in rows.itertuples()
    ]
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Delta P(good-side final answer): retain - divergent resample")
    axis.set_title("Which reasoning sentences causally control the answer?", fontweight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _finish(fig, destination)


def plot_lens_heatmap(records: pd.DataFrame, destination: str | Path) -> Path:
    """Plot signed direction contrasts over layer and named position for J/R lenses."""

    required = {"lens_type", "layer", "position", "concept_set", "signed_contrast"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"lens records missing columns: {sorted(missing)}")
    subset = records[records["concept_set"] == "direction"].copy()
    if subset.empty:
        raise ValueError("no direction lens records")
    position_order = [
        "prompt_end",
        "first_estimate_pre",
        "anchor_pre",
        "anchor_post",
        "final_answer_pre",
    ]
    lens_types = [kind for kind in ["j", "r"] if kind in set(subset["lens_type"])]
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(lens_types), figsize=(6.2 * len(lens_types), 6), squeeze=False)
    limit = float(np.nanpercentile(np.abs(subset["signed_contrast"]), 98)) or 1.0
    for axis, lens_type in zip(axes[0], lens_types, strict=True):
        table = (
            subset[subset["lens_type"] == lens_type]
            .groupby(["layer", "position"], observed=True)["signed_contrast"]
            .mean()
            .unstack("position")
            .reindex(columns=position_order)
            .sort_index()
        )
        image = axis.imshow(
            table.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xticks(
            np.arange(len(table.columns)), [p.replace("_", "\n") for p in table.columns]
        )
        layer_ticks = np.linspace(0, len(table.index) - 1, min(7, len(table.index)), dtype=int)
        axis.set_yticks(layer_ticks, [str(table.index[i]) for i in layer_ticks])
        axis.set_ylabel("Layer")
        axis.set_title(f"{lens_type.upper()}-lens")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Signed direction contrast")
    fig.suptitle(
        "Is the good-side direction represented before it is verbalized?", fontweight="bold"
    )
    fig.tight_layout()
    return _finish(fig, destination)
