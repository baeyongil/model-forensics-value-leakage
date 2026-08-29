from __future__ import annotations

from pathlib import Path

import pandas as pd

import model_forensics.figures as figures


def _behavior_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "giraffe",
                "condition": condition,
                "stage": stage,
                "rate": rate,
                "ci_low": max(0.0, rate - 0.1),
                "ci_high": min(1.0, rate + 0.1),
            }
            for condition, rates in (
                ("baseline", (0.4, 0.5)),
                ("threshold_only", (0.5, 0.6)),
                ("above_good", (0.6, 0.7)),
                ("below_good", (0.7, 0.4)),
            )
            for stage, rate in zip(("first", "final"), rates, strict=True)
        ]
    )


def test_first_vs_final_figure_exactly_labels_above_threshold_metric(
    monkeypatch,
) -> None:
    captured = {}

    def capture(figure, destination: str | Path) -> Path:
        captured["figure"] = figure
        return Path(destination)

    monkeypatch.setattr(figures, "_finish", capture)
    figures.plot_first_vs_final_bias(_behavior_rows(), "unused.png")

    figure = captured["figure"]
    assert figure._suptitle.get_text() == "When do estimates cross the numeric threshold?"
    assert figure.axes[0].get_ylabel() == "Probability estimate exceeds the threshold"
    assert "good-side" not in figure._suptitle.get_text().lower()


def test_sentence_effect_forest_renders_legitimate_nonestimable_rows(
    monkeypatch,
) -> None:
    captured = {}

    def capture(figure, destination: str | Path) -> Path:
        captured["figure"] = figure
        return Path(destination)

    monkeypatch.setattr(figures, "_finish", capture)
    effects = pd.DataFrame(
        [
            {
                "sentence_class": "accuracy_commitment",
                "direction": "pooled",
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "conclusion": "inconclusive",
            },
            {
                "sentence_class": "value_threshold_planning",
                "direction": "pooled",
                "estimate": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
                "conclusion": "positive",
            },
        ]
    )

    figures.plot_sentence_effect_forest(effects, "unused.png")

    axis = captured["figure"].axes[0]
    assert any(text.get_text() == "NA (not estimable)" for text in axis.texts)
    assert any("[not estimable]" in label.get_text() for label in axis.get_yticklabels())
