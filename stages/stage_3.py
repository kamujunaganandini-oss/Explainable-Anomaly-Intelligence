def rank_causal_hypotheses(context, hypothesis_library):
    """
    Computes posterior probabilities for competing causal explanations.
    
    Parameters:
    - context: Enriched anomaly context
    - hypothesis_library: Dict of hypotheses with prior probabilities and likelihood functions
    
    Returns:
    - Ranked list of hypotheses with posterior probabilities
    """
    posteriors = []
    for hyp_name, hyp_config in hypothesis_library.items():
        prior = hyp_config['prior']
        likelihood = hyp_config['likelihood_fn'](context)
        posterior = prior * likelihood
        posteriors.append({'hypothesis': hyp_name, 'posterior': posterior})
    
    # Normalize
    total = sum([p['posterior'] for p in posteriors])
    for p in posteriors:
        p['posterior'] /= total
    
    return sorted(posteriors, key=lambda x: x['posterior'], reverse=True)