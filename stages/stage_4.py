# stages/stage_4.py
'''
#temporary mock up
def generate_narrative(top_hypothesis, context):
    """
    Mock narrative generator (LLM-free).

    Parameters:
    - top_hypothesis: dict with keys ['hypothesis', 'evidence']
    - context: dict with keys ['metric', 'magnitude', 'date']

    Returns:
    - Executive-friendly narrative string
    """

    return f"""
EXECUTIVE SUMMARY

An anomaly was detected on {context['date']} affecting:
{context['metric']} (deviation magnitude: {context['magnitude']}).

Most likely cause:
• {top_hypothesis['hypothesis']}

Supporting evidence:
• {top_hypothesis['evidence']}

Recommended next step:
• Validate this hypothesis with domain stakeholders and
  review recent operational or data changes.
"""

'''
##anthropic incase credits


import anthropic

def generate_narrative(top_hypothesis, context, run_stats):
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
    This analysis run processed {run_stats['total_rows']} records.

    Total anomalies detected before filtering: {run_stats['raw_anomalies']}.
    Total high-severity anomalies analyzed: {run_stats['significant_anomalies']}.

    Anomaly detected:
    • Metric: {context['metric']}
    • Deviation: {context['magnitude']}%
    • Date: {context['date']}

    Most likely cause:
    • {top_hypothesis['hypothesis']}

    Supporting evidence:
    • {top_hypothesis['evidence']}

    Generate a 3-sentence executive summary that:
    1. States what happened
    2. Explains the probable cause
    3. Recommends an action

    If the number of raw anomalies is low, emphasize monitoring.
    If the number of raw anomalies is high, emphasize urgency and systemic investigation.

    Use direct, non-technical language suitable for senior leadership.
    """
    '''prompt = f"""
    Anomaly detected: {context['metric']} deviated {context['magnitude']}% on {context['date']}.
    Most likely cause: {top_hypothesis['hypothesis']}
    Supporting evidence: {top_hypothesis['evidence']}
    
    Generate a 3-sentence executive summary that:
    1. States what happened
    2. Explains the probable cause
    3. Recommends an action
    
    Use direct, non-technical language suitable for senior leadership.
    """
    '''
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
