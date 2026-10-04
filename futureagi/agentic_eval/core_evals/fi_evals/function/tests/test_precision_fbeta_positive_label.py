"""Regression tests for positive-class resolution in precision / F-beta.

Both evaluators used to auto-select the positive class alphabetically, which
resolves to the *negative* member of every common binary convention (no, false,
0, negative, fail, ham). These tests pin the corrected behaviour.
"""

from __future__ import annotations

import math

import pytest

from agentic_eval.core_evals.fi_evals.function.functions import (
    calculate_f_beta_score,
    calculate_precision_score,
)


def _approx(actual, expected, rel=1e-9, abs_=1e-12):
    """Float comparison with tight tolerances."""
    assert math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_), (
        f"{actual!r} != {expected!r}"
    )


# A spam classifier over 10 samples: 2 truly spam, 3 predicted spam, 2 correct.
# Precision for "spam" is 2/3; precision for "ham" is 7/7 = 1.0.
SPAM_LABELS = ["spam", "spam", "ham", "ham", "ham", "ham", "ham", "ham", "ham", "ham"]
SPAM_PREDS = ["spam", "spam", "spam", "ham", "ham", "ham", "ham", "ham", "ham", "ham"]


class TestConventionalPositiveLabel:
    """Auto-detection must resolve to the positive member of a known pair."""

    def test_spam_ham_does_not_score_the_negative_class(self):
        # Regression: previously returned 1.0, the precision of "ham".
        result = calculate_precision_score(SPAM_PREDS, SPAM_LABELS)
        _approx(result["result"], 2 / 3)
        assert "spam" in result["reason"]

    def test_spam_ham_f_beta_scores_spam(self):
        result = calculate_f_beta_score(SPAM_PREDS, SPAM_LABELS)
        _approx(result["result"], 0.8)
        assert "spam" in result["reason"]

    def test_auto_matches_explicit_positive_label(self):
        auto = calculate_precision_score(SPAM_PREDS, SPAM_LABELS)
        explicit = calculate_precision_score(
            SPAM_PREDS, SPAM_LABELS, positive_label="spam"
        )
        _approx(auto["result"], explicit["result"])

    @pytest.mark.parametrize(
        "positive,negative",
        [
            ("yes", "no"),
            ("true", "false"),
            ("1", "0"),
            ("positive", "negative"),
            ("pass", "fail"),
            ("spam", "ham"),
            ("relevant", "irrelevant"),
            ("correct", "incorrect"),
            ("valid", "invalid"),
        ],
    )
    def test_every_convention_resolves_to_the_positive_member(self, positive, negative):
        labels = [positive, positive, negative, negative]
        preds = [positive, positive, positive, negative]

        # For the positive class: TP=2, FP=1, FN=0 -> P=2/3, R=1.0, F1=0.8.
        # For the negative class: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=2/3.
        precision = calculate_precision_score(preds, labels)
        assert math.isclose(precision["result"], 2 / 3), (
            f"precision scored the wrong class for {positive}/{negative}"
        )
        assert f"positive='{positive}'" in precision["reason"]

        f_beta = calculate_f_beta_score(preds, labels)
        assert math.isclose(f_beta["result"], 0.8), (
            f"f_beta scored the wrong class for {positive}/{negative}"
        )
        assert f"positive='{positive}'" in f_beta["reason"]

    def test_convention_is_case_insensitive(self):
        labels = ["Yes", "Yes", "No", "No"]
        preds = ["Yes", "Yes", "Yes", "No"]
        _approx(calculate_precision_score(preds, labels)["result"], 2 / 3)

    def test_convention_resolves_from_union_of_labels_and_predictions(self):
        # "no" never appears in ground truth but is predicted; the pair is still
        # recognised, so precision is scored for "yes" (2 of 3 correct).
        labels = ["yes", "yes", "yes"]
        preds = ["yes", "yes", "no"]
        result = calculate_precision_score(preds, labels)
        _approx(result["result"], 1.0)
        assert "positive='yes'" in result["reason"]


class TestExplicitPositiveLabel:
    def test_explicit_label_overrides_convention(self):
        result = calculate_precision_score(
            SPAM_PREDS, SPAM_LABELS, positive_label="ham"
        )
        _approx(result["result"], 1.0)

    def test_falsy_integer_zero_is_honoured_by_precision(self):
        labels = ["0", "0", "1", "1"]
        preds = ["0", "0", "0", "1"]
        # positive_label=0 is falsy; it must not be discarded.
        result = calculate_precision_score(preds, labels, positive_label=0)
        _approx(result["result"], 2 / 3)
        assert "positive='0'" in result["reason"]

    def test_falsy_integer_zero_is_honoured_by_f_beta(self):
        labels = ["0", "0", "1", "1"]
        preds = ["0", "0", "0", "1"]
        result = calculate_f_beta_score(preds, labels, positive_label=0)
        # P = 2/3, R = 1.0 -> F1 = 0.8
        _approx(result["result"], 0.8)
        assert "positive='0'" in result["reason"]

    @pytest.mark.parametrize(
        "scorer, expected",
        [(calculate_precision_score, 1.0), (calculate_f_beta_score, 2 / 3)],
    )
    def test_falsy_zero_with_a_lower_sorting_class(self, scorer, expected):
        # Sentiment labels -1 / 0 / 1, scoring the neutral class 0. "-1" sorts
        # ahead of "0", so the old truthiness gate discarded positive_label=0
        # and silently scored "-1" instead, reporting a perfect 1.0 for F1.
        labels = ["0", "0", "1", "-1", "1"]
        preds = ["0", "1", "1", "-1", "1"]
        result = scorer(preds, labels, positive_label=0)
        _approx(result["result"], expected)
        assert "positive='0'" in result["reason"]

    def test_int_and_str_positive_label_agree(self):
        labels = ["0", "0", "1", "-1", "1"]
        preds = ["0", "1", "1", "-1", "1"]
        as_int = calculate_f_beta_score(preds, labels, positive_label=0)["result"]
        as_str = calculate_f_beta_score(preds, labels, positive_label="0")["result"]
        _approx(as_int, as_str)

    def test_falsy_string_zero_is_honoured(self):
        labels = ["0", "0", "1", "1"]
        preds = ["0", "0", "0", "1"]
        _approx(
            calculate_f_beta_score(preds, labels, positive_label="0")["result"],
            0.8,
        )
