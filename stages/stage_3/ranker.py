# stages/stage_3/ranker.py

from .hypotheses import HYPOTHESES
from .likelihoods import (
    demand_spike_likelihood,
    supply_disruption_likelihood,
    data_quality_likelihood
)

LIKELIHOOD_MAP = {
    "demand_spike": demand_spike_likelihood,
    "supply_disruption": supply_disruption_likelihood,
    "data_quality_issue": data_quality_likelihood
}


def rank_hypotheses(context):
    results = []

    for h in HYPOTHESES:
        name = h["name"]
        prior = h["prior"]

        likelihood_fn = LIKELIHOOD_MAP[name]
        likelihood = likelihood_fn(context)

        posterior = prior * likelihood

        results.append({
            "hypothesis": name,
            "prior": prior,
            "likelihood": round(likelihood, 3),
            "posterior": posterior
        })

    # Normalize posteriors
    total = sum(r["posterior"] for r in results)
    for r in results:
        r["posterior"] = round(r["posterior"] / total, 3) if total > 0 else 0

    # Sort descending
    results.sort(key=lambda x: x["posterior"], reverse=True)

    return results
