def run_stage4_v2(
    stage1_output: dict,
    stage2_output: dict | None,
    stage3_output: list | None
):
    """
    CADEN Stage 4 v2
    Human-readable narrative generation
    """

    risk = stage1_output["risk_score"]
    anomaly = stage1_output["anomaly_level"]
    trend = stage1_output["trend"]

    # -----------------------------
    # 1. Situation summary
    # -----------------------------
    if anomaly in ["none", "marginal"]:
        summary = (
            "No significant anomaly detected in the recent period. "
            "System behavior remains within expected bounds."
        )
    else:
        summary = (
            f"A {anomaly} anomaly was detected over the recent period, "
            "indicating notable deviation from normal system behavior."
        )

    # -----------------------------
    # 2. Key observations
    # -----------------------------
    observations = []

    if stage2_output:
        for feature, ctx in stage2_output.get("temporal_context", {}).items():
            shift = ctx["mean_shift"]
            direction = "increased" if shift > 0 else "decreased"
            observations.append(
                f"{feature.replace('_', ' ').title()} {direction} compared to the prior period."
            )

        for feature, ctx in stage2_output.get("directional_context", {}).items():
            if ctx["down_days"] > ctx["up_days"]:
                observations.append(
                    f"{feature.replace('_', ' ').title()} showed a consistent downward trend."
                )
            else:
                observations.append(
                    f"{feature.replace('_', ' ').title()} showed a consistent upward trend."
                )

        for feature, ctx in stage2_output.get("variability_context", {}).items():
            observations.append(
                f"{feature.replace('_', ' ').title()} became more unstable during this period."
            )

        for pair, corr in stage2_output.get("relational_context", {}).items():
            f1, f2 = pair.split("__")
            observations.append(
                f"{f1.replace('_', ' ').title()} and {f2.replace('_', ' ').title()} moved together during this period."
            )

    if not observations:
        observations.append("No notable changes were observed in system behavior.")

    # -----------------------------
    # 3. Likely explanations
    # -----------------------------
    explanations = []

    if stage3_output:
        top = stage3_output[0]
        explanations.append(
            f"The most likely explanation is '{top['description']}'."
        )

        if len(stage3_output) > 1 and stage3_output[1]["confidence"] > 0.2:
            second = stage3_output[1]
            explanations.append(
                f"A secondary contributing factor may be '{second['description']}'."
            )
    else:
        explanations.append(
            "No specific explanation is suggested as system behavior remains within normal limits."
        )

    # -----------------------------
    # 4. Confidence & urgency
    # -----------------------------
    if trend in ["sharp_increase", "increasing"]:
        urgency = "elevated"
    else:
        urgency = "low"

    if risk < 0.3:
        confidence = "high confidence in system stability"
    elif risk < 0.6:
        confidence = "moderate confidence in identified patterns"
    else:
        confidence = "high confidence in anomaly detection"

    # -----------------------------
    # Output narrative
    # -----------------------------
    return {
        "summary": summary,
        "observations": observations,
        "explanations": explanations,
        "confidence": confidence,
        "urgency": urgency
    }
