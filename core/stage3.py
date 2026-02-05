def compute_likelihood(stage2_output, expectations):
    """
    Generic likelihood computation based on signal match
    """
    signals = stage2_output.get("signals", {})
    score = 1.0
    matched = 0

    for feature, expected_dir in expectations.items():
        observed = signals.get(feature)

        if not observed:
            continue

        if observed["direction"] == expected_dir:
            score *= observed["strength"]
            matched += 1
        else:
            score *= 0.2  # soft penalty

    if matched == 0:
        return 0.1  # weak generic likelihood

    return max(score, 0.05)



def run_stage3_v2(stage2_output, stage1_output, hypotheses_config):
    """
    Stage 3 v2
    Bayesian causal hypothesis ranking (v1 semantics preserved)
    """

    results = []

    for hyp_name, hyp_cfg in hypotheses_config.items():
        prior = hyp_cfg["prior"]
        expectations = hyp_cfg.get("expectations", {})

        likelihood = compute_likelihood(stage2_output, expectations)
        posterior = prior * likelihood

        results.append({
            "hypothesis": hyp_name,
            "prior": prior,
            "posterior": posterior
        })

    # Normalize posteriors
    total = sum(r["posterior"] for r in results)
    if total > 0:
        for r in results:
            r["posterior"] /= total
    else:
        n = len(results)
        for r in results:
            r["posterior"] = 1 / n

    # Rank
    results.sort(key=lambda x: x["posterior"], reverse=True)

    return results
