import pandas as pd
import yaml

from core.state_builder import build_system_state
from core.stage1 import run_stage1_v2
from core.stage2 import run_stage2_v2
from core.stage3 import run_stage3_v2
from core.stage4 import run_stage4_llm
from core.stage5 import compute_nci, compute_der, compute_cds


with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load raw data
df_raw = pd.read_csv(config["data"]["path"])

# Build system state 
state_df = build_system_state(
    df_raw=df_raw,
    date_column=config["state_builder"]["date_column"],
    time_unit=config["state_builder"]["time_unit"],
    aggregation_config=config["state_builder"]["aggregations"],
    date_format=config["state_builder"].get("date_format")
)

# Stage 1 v2
stage1_out = run_stage1_v2(
    df=state_df,
    time_column="time",
    feature_list=config["stage1"]["features"],
    feature_bounds=config["stage1"]["feature_bounds"],
    window_days=config["stage1"]["window_days"],
    alpha=config["stage1"]["alpha"],
    weights=config["stage1"]["risk_weights"]
)


print(stage1_out)
print("\n--- Conditional Routing ---")

anomaly_level = stage1_out.get("anomaly_level")
decision_gate = stage1_out.get("decision_gate")

# -----------------------------
# Case 1: No or Marginal Anomaly
# -----------------------------
if decision_gate == "stop":

    print(f"No deep analysis required (anomaly level: {anomaly_level}).")

    stage4_out = run_stage4_llm(
        stage1_output=stage1_out,
        stage2_output=None,
        stage3_output=None
    )

    print("\nStage 4 Summary:")
    print(stage4_out)

    print("\nStage 5 Metrics:")
    print("NCI: N/A")
    print("DER: N/A")
    print("CDS: N/A")

# -----------------------------
# Case 2: Moderate / Strong Anomaly
# -----------------------------
else:

    print(f"Anomaly detected (level: {anomaly_level}). Running full pipeline.")

    # ---- Stage 2 ----
    stage2_out = run_stage2_v2(
        state_df=state_df,
        stage1_output=stage1_out,
        features=config["stage1"]["features"],
        baseline_days=config["stage1"]["window_days"]
    )

    print("\nStage 2 Output:")
    print(stage2_out)

    # ---- Stage 3 ----
    stage3_out = run_stage3_v2(
        stage2_output=stage2_out,
        stage1_output=stage1_out,
        hypotheses_config=config["stage3"]["hypotheses"]
    )

    print("\nStage 3 Output:")
    print(stage3_out)

    # ---- Stage 4 ----
    stage4_out = run_stage4_llm(
        stage1_output=stage1_out,
        stage2_output=stage2_out,
        stage3_output=stage3_out
    )

    print("\nStage 4 Summary:")
    print(stage4_out)

    # ---- Stage 5 ----
    prior_probs = [h["prior"] for h in stage3_out]
    post_probs = [h["posterior"] for h in stage3_out]

    causal_influences = {
        h["hypothesis"]: h["posterior"]
        for h in stage3_out
    }

    nci = compute_nci(stage3_out)
    der = compute_der(prior_probs, post_probs)
    cds = compute_cds(causal_influences)

    print("\nStage 5 Metrics:")
    print("NCI:", round(nci, 3))
    print("DER:", round(der, 3))
    print("CDS:", round(cds, 3))
