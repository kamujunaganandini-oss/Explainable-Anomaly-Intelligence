import pandas as pd
from utils.config import load_config
from stages.stage_1 import detect_multivariate_anomalies
from stages.stage_2.context_builder import build_context
from stages.stage_3.ranker import rank_hypotheses
from stages.stage_4 import generate_narrative
from stages.stage_5 import compute_nci, compute_der, compute_cds



def run():
    config = load_config()

    df = pd.read_csv(config["data"]["path"])
    features = config["data"]["features"]
    alpha = config["anomaly_detection"]["alpha"]

    result = detect_multivariate_anomalies(df,features,alpha)
    
    print("Stage 1 complete")
    print("Total Rows : ", len(result))
    print("Anomalies detected:", result["is_anomaly"].sum())
    print(result[["date", "T2_score", "is_anomaly"]])
    anomalies = result[result["is_anomaly"]]
    
    ##stage_2
    anomalies = result[result["is_anomaly"]]

    for row in anomalies.itertuples():
        context = build_context(
            anomaly_row=row,
            df=df,
            feature_cols=features,
            config=config
        )
        print("\n Enriched Context")
        print(context)
    ##stage 3
    ranked = rank_hypotheses(context)

    print("\n🧠 Hypothesis Ranking")
    for r in ranked:
        print(r)

    ##STAGE 4
    # ===== Stage 4: Narrative Generation =====
    anomaly = context["anomaly"]
    top_hypothesis = ranked[0]

    stage_4_context = {
        "metric": ", ".join(anomaly["features"].keys()),
        "magnitude": round(anomaly["t2_score"], 2),
        "date": anomaly["date"]
    }
    #STAGE_4_EVIDENCE 
    def build_evidence(context):
        deviations = context["relational"]["feature_deviations"]

        signals = []
        for metric, info in deviations.items():
            signals.append(
                f"{metric} {info['direction']} (z={round(info['z_score'],2)})"
            )

        return "; ".join(signals)


    stage_4_hypothesis = {
        "hypothesis": top_hypothesis["hypothesis"],
        "evidence": build_evidence(context)
    }


    try:
        top_hypothesis = ranked[0]

        narrative = generate_narrative(
            top_hypothesis=stage_4_hypothesis,
            context=stage_4_context
        )

        print("\n Stage 4 Narrative")
        print(narrative)

    except Exception as e:
        print("\n Stage 4 failed")
        print(str(e))


    ##STAGE 5
    # Extract inputs for Stage 5
    prior_action_probs = [h["prior"] for h in ranked]
    post_action_probs = [h["posterior"] for h in ranked]

    # Simple causal influence proxy (v1)
    causal_influences = {
        h["hypothesis"]: h["posterior"] for h in ranked
    }
    
    nci = compute_nci(ranked)
    der = compute_der(prior_action_probs, post_action_probs)
    cds = compute_cds(causal_influences)
    print("\n Stage 5 Metrics")
    print("NCI:", round(nci, 3))
    print("DER:", round(der, 3))
    print("CDS:", round(cds, 3))


if __name__ == "__main__":
    run()