import numpy as np

def compute_nci(posteriors):
    """Narrative Confidence Index: Max posterior probability"""
    return max([p['posterior'] for p in posteriors])

def compute_der(prior_action_probs, post_action_probs):
    """Decision Entropy Reduction"""
    H_before = -sum([p * np.log2(p) for p in prior_action_probs if p > 0])
    H_after = -sum([p * np.log2(p) for p in post_action_probs if p > 0])
    return H_before - H_after

def compute_cds(causal_influences):
    """Causal Directedness Score"""
    total_influence = sum(causal_influences.values())
    normalized = [v / total_influence for v in causal_influences.values()]
    entropy = -sum([p * np.log2(p) for p in normalized if p > 0])
    max_entropy = np.log2(len(causal_influences))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0

'''
```

---

## Architecture: Decision Intelligence at Scale

The CADEN framework is deployed as a microservices architecture with three core layers:

### Layer 1: Data Ingestion & Preprocessing
- **Real-time connectors** to enterprise data warehouses (e.g., parts inventory, repair orders, supplier transactions)
- **Time-series normalization** to handle seasonality, business calendar effects, and missing data
- **Feature engineering** to construct multivariate monitoring vectors

### Layer 2: Anomaly Intelligence Engine
- **Detection service** runs Hotelling's T² continuously on rolling windows
- **Context retrieval service** queries metadata stores for enrichment
- **Hypothesis engine** applies Bayesian inference to rank causal explanations
- **Narrative generator** interfaces with LLM API to produce natural language summaries

### Layer 3: Executive Dashboard (Flask-based UI)

The dashboard presents:

- **KPI Tiles**: NCI, DER, CDS displayed as color-coded risk zones
- **Anomaly Feed**: Chronological list of detected anomalies with one-line summaries
- **Deep Dive View**: Full narrative, supporting evidence, recommended actions
- **Impact Projection**: Estimated financial or operational consequences if no action is taken

The UI is designed for non-technical stakeholders. No charts, no statistical jargon—only decision-ready insights.

**Sample Dashboard Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Anomaly Intelligence Dashboard                              │
├─────────────────────────────────────────────────────────────┤
│  [NCI: 0.82 ●] [DER: 1.3 bits ●] [CDS: 0.71 ●]              │
│  Status: 2 High-Confidence Anomalies Require Action         │
├─────────────────────────────────────────────────────────────┤
│  📌 Northeast Region RO Volume Drop (-12%)                   │
│  Cause: Interstate Batteries shipment delay                  │
│  Impact: $240K Q2 revenue shortfall                          │
│  Action: Expedite backlog | Notify service advisors          │
│  [View Details]                                              │
├─────────────────────────────────────────────────────────────┤
│  📌 Midwest Warranty Claims Spike (+18%)                     │
│  Cause: New part family defect pattern emerging              │
│  Impact: Potential recall exposure                           │
│  Action: Quality audit | Supplier investigation              │
│  [View Details]                                              │
└─────────────────────────────────────────────────────────────┘

'''