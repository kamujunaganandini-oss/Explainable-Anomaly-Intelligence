import numpy as np

def compute_nci(posteriors):
    return max([p["posterior"] for p in posteriors])

def compute_der(prior_action_probs, post_action_probs):
    H_before = -sum([p * np.log2(p) for p in prior_action_probs if p > 0])
    H_after = -sum([p * np.log2(p) for p in post_action_probs if p > 0])
    return H_before - H_after

def compute_cds(causal_influences):
    total = sum(causal_influences.values())
    normalized = [v / total for v in causal_influences.values()]
    entropy = -sum([p * np.log2(p) for p in normalized if p > 0])
    max_entropy = np.log2(len(causal_influences))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0
