"""Statistical hypothesis testing (credit card vs. cash base fares)."""
from __future__ import annotations

import pandas as pd
import scipy.stats as scipy_stats

__all__ = ["welch_ttest"]


def welch_ttest(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Welch's two-sample t-test: mean fare_amount, credit card (1) vs cash (2)."""
    # 1. Isolate the fare amounts for credit card (1) and cash (2) payments
    card_fares = df[df['payment_type'] == 1]['fare_amount']
    cash_fares = df[df['payment_type'] == 2]['fare_amount']

    print(f"--- Sample Sizes and Descriptive Statistics ---")
    print(f"Credit Card: {len(card_fares):,} trips | Mean Fare: ${card_fares.mean():.2f} | Std Dev: ${card_fares.std():.2f}")
    print(f"Cash:        {len(cash_fares):,} trips | Mean Fare: ${cash_fares.mean():.2f} | Std Dev: ${cash_fares.std():.2f}")

    # 2. Welch's t-test (does not assume equal variances)
    t_stat, p_val = scipy_stats.ttest_ind(card_fares, cash_fares, equal_var=False)

    print(f"\n--- Welch's Two-Sample T-test Results ---")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_val:.4e}")

    # 3. Decision rule
    if p_val < alpha:
        print(f"\nDecision: Reject the Null Hypothesis (p < 0.05). There is a statistically significant difference in fare amounts between credit card and cash payments.")
    else:
        print(f"\nDecision: Fail to Reject the Null Hypothesis (p >= 0.05). No statistically significant difference was observed.")
    return {"t_stat": t_stat, "p_value": p_val,
            "card_mean": card_fares.mean(), "cash_mean": cash_fares.mean()}
