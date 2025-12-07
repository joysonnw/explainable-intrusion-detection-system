from typing import List, Dict, Any

BASIC_GUIDANCE = {
    "BENIGN": "Traffic appears normal. Monitor trends; no immediate action.",
    "DoS Hulk": "High request rate with short intervals. Rate-limit, check WAF/IPS.",
    "PortScan": "Multiple connection attempts across ports. Inspect source IPs; block or throttle.",
    "DDoS": "Distributed high-volume traffic. Engage mitigation, upstream filtering, and CDN shields.",
    "Heartbleed": "Potential OpenSSL heartbeat exploit. Patch OpenSSL and rotate credentials.",
    "Brute Force": "Repeated auth failures. Enforce lockouts and MFA; audit logs.",
    "Web Attack": "Anomalous HTTP patterns. Review WAF rules and web server logs.",
}

def _resolve_name(label_id: int, class_names: List[str]) -> str:
    if 0 <= label_id < len(class_names):
        return class_names[label_id]
    return str(label_id)

def get_actionable_insight(pred_label: int,
                           shap_payload: Dict[str, Any],
                           class_names: List[str]) -> str:
    name = _resolve_name(pred_label, class_names)
    guidance = BASIC_GUIDANCE.get(name, "Suspicious activity detected. Review logs around this flow and isolate the source if needed.")

    try:
        feats = shap_payload["feature_names"]
        vals_list = shap_payload["values"]
        idx = pred_label if pred_label < len(vals_list) else 0
        vals = vals_list[idx]
        abs_contrib = sorted(
            zip(feats, map(abs, vals[0] if (hasattr(vals, 'ndim') and vals.ndim > 1) else vals)),
            key=lambda t: t[1],
            reverse=True
        )[:5]
        top_feats = ", ".join(f for f, _ in abs_contrib)
        return f"{guidance} Top contributing features: {top_feats}."
    except Exception:
        return guidance
