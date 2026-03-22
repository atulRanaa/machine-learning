"""
Lab: ML Systems - Drift Detection, Model Serving, and Fairness Audit
=====================================================================
"""
import numpy as np
from scipy import stats


# =============================================================================
# DRIFT DETECTION TOOLKIT
# =============================================================================
def population_stability_index(expected, actual, bins=10):
    """Compute PSI to detect data distribution shift."""
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pcts = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pcts = np.histogram(actual, breakpoints)[0] / len(actual)
    expected_pcts = np.clip(expected_pcts, 1e-4, None)
    actual_pcts = np.clip(actual_pcts, 1e-4, None)
    return np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))


def detect_drift(reference, production, alpha=0.05):
    """Multi-test drift detection: KS test + PSI."""
    ks_stat, p_value = stats.ks_2samp(reference, production)
    psi = population_stability_index(reference, production)
    return {
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 6),
        "drift_detected_ks": p_value < alpha,
        "psi": round(psi, 4),
        "drift_detected_psi": psi > 0.2,  # PSI > 0.2 = significant drift
    }


# =============================================================================
# FAIRNESS AUDIT
# =============================================================================
def fairness_audit(y_true, y_pred, sensitive_attr):
    """Compute fairness metrics across groups defined by sensitive_attr."""
    groups = np.unique(sensitive_attr)
    results = {}

    for g in groups:
        mask = sensitive_attr == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        tp = np.sum((y_p == 1) & (y_t == 1))
        fp = np.sum((y_p == 1) & (y_t == 0))
        fn = np.sum((y_p == 0) & (y_t == 1))
        tn = np.sum((y_p == 0) & (y_t == 0))

        results[f"group_{g}"] = {
            "positive_rate": round(np.mean(y_p == 1), 4),
            "tpr": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
            "accuracy": round(np.mean(y_p == y_t), 4),
            "count": int(mask.sum()),
        }

    # Demographic parity: difference in positive rates
    rates = [results[f"group_{g}"]["positive_rate"] for g in groups]
    results["demographic_parity_diff"] = round(max(rates) - min(rates), 4)

    # Equalized odds: difference in TPR and FPR
    tprs = [results[f"group_{g}"]["tpr"] for g in groups]
    fprs = [results[f"group_{g}"]["fpr"] for g in groups]
    results["equalized_odds_tpr_diff"] = round(max(tprs) - min(tprs), 4)
    results["equalized_odds_fpr_diff"] = round(max(fprs) - min(fprs), 4)

    return results


if __name__ == "__main__":
    np.random.seed(42)

    # --- Drift Detection Demo ---
    print("=" * 60)
    print("DRIFT DETECTION")
    print("=" * 60)
    reference = np.random.normal(0, 1, 1000)
    no_drift = np.random.normal(0, 1, 1000)
    with_drift = np.random.normal(0.5, 1.2, 1000)

    print("No drift:", detect_drift(reference, no_drift))
    print("With drift:", detect_drift(reference, with_drift))

    # --- Fairness Audit Demo ---
    print("\n" + "=" * 60)
    print("FAIRNESS AUDIT")
    print("=" * 60)
    n = 1000
    sensitive = np.random.choice([0, 1], n)
    y_true = np.random.choice([0, 1], n)
    # Simulate biased predictions: higher positive rate for group 1
    y_pred = y_true.copy()
    bias_mask = (sensitive == 0) & (np.random.random(n) < 0.3)
    y_pred[bias_mask] = 0  # Suppress positives for group 0

    audit = fairness_audit(y_true, y_pred, sensitive)
    for key, val in audit.items():
        print(f"  {key}: {val}")
