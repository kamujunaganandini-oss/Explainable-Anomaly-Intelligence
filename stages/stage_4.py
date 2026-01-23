import anthropic

def generate_narrative(top_hypothesis, context):
    """
    Generates executive-friendly narrative using Claude API.
    
    Parameters:
    - top_hypothesis: Highest-ranked causal explanation
    - context: Enriched anomaly context
    
    Returns:
    - Structured narrative string
    """
    client = anthropic.Anthropic()
    
    prompt = f"""
    Anomaly detected: {context['metric']} deviated {context['magnitude']}% on {context['date']}.
    Most likely cause: {top_hypothesis['hypothesis']}
    Supporting evidence: {top_hypothesis['evidence']}
    
    Generate a 3-sentence executive summary that:
    1. States what happened
    2. Explains the probable cause
    3. Recommends an action
    
    Use direct, non-technical language suitable for senior leadership.
    """
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text