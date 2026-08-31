import numpy as np


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def linear_fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "stage": fit["stage"],
        "label": fit["label"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }

    for key in ["A", "A_err", "B", "B_err", "C", "C_err", "D", "D_err", "L", "L_err"]:
        if key in fit:
            out[key] = fit[key]

    if "basis_terms" in fit:
        out["basis_terms"] = fit["basis_terms"]
        out["coeffs"] = fit["coeffs"]
        out["coeff_errs"] = fit["coeff_errs"]

    return to_serializable(out)


def physical_fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "stage": fit["stage"],
        "label": fit["label"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }

    for key in [
        "m_M_chi_sq",
        "m_M_chi_sq_err",
        "L_m_M",
        "L_m_M_err",
        "Q_m_M",
        "Q_m_M_err",
        "W_m_M",
        "W_m_M_err",
        "R_m_M",
        "R_m_M_err",
        "C_m_M",
        "C_m_M_err",
    ]:
        if key in fit:
            out[key] = fit[key]

    if "fix_Q_to_zero" in fit:
        out["fix_Q_to_zero"] = fit["fix_Q_to_zero"]

    return to_serializable(out)
