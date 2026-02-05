import os
import anthropic


def run_stage4_llm(stage1_output, stage2_output=None, stage3_output=None):
    """
    Stage 4: Narrative generation
    - Informational narrative when no anomaly
    - LLM narrative only when explanation exists
    """
    top = stage3_output[0]

    metric_summary = ", ".join(
        [k for k, v in stage2_output["signals"].items() if v["strength"] > 0.3]
    )

    magnitude = round(stage1_output["risk_score"] * 100, 1)
    date = f"{stage1_output['window_start']} to {stage1_output['window_end']}"

    evidence = "; ".join(
        f"{k} {v['direction']}"
        for k, v in stage2_output["signals"].items()
        if v["strength"] > 0.3
    )

    # -----------------------------
    # CASE 1: No or marginal anomaly
    # -----------------------------
    if stage1_output.get("decision_gate") == "stop":
        reason = stage1_output.get("reason", "No significant anomaly detected")

        return (
            "System Status Update:\n\n"
            "No significant anomalies were detected during the current monitoring window.\n"
            f"Reason: {reason}.\n"
            "The system remains within expected operational behavior. No action is required."
        )

    # -----------------------------
    # CASE 2: Anomaly present
    # -----------------------------
    hypothesis_name = "Unknown"
    confidence = "N/A"

    if stage3_output and len(stage3_output) > 0:
        hypothesis_name = stage3_output[0].get("hypothesis", "Unknown")
        confidence = round(stage3_output[0].get("posterior", 0.0), 2)

    prompt = f"""
    Anomaly detected: {metric_summary} deviated during {date}.
    Severity level: {stage1_output['anomaly_level']} ({magnitude}% risk score).

    Most likely cause: {top['hypothesis']} ({top['posterior']:.0%} confidence)
    Supporting evidence: {evidence}

    Generate a 3-sentence executive summary that:
    1. States what happened
    2. Explains the probable cause
    3. Recommends an action

    Use direct, non-technical language suitable for senior leadership.
    """


    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Anthropic API key not found")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text.strip()

    except Exception as e:
        print("\n[Stage 4 LLM Exception]")
        print(type(e).__name__, ":", str(e))

        return (
            "An anomaly was detected, but an automated narrative could not be generated.\n"
            f"Top hypothesis: {hypothesis_name}\n"
            "Please refer to structured analysis and KPIs."
        )
