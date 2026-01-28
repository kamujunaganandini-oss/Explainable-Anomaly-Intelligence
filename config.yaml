usecase:
  name: "Product Release Rollback Decision"

data:
  path: "data/product_release_100k.csv"
  time_column: "day"

stage1:
  features:
    - dau
    - session_duration
    - conversion_rate
    - error_rate
    - api_latency_ms
    - crash_rate
    - support_tickets
    - feature_adoption
    - page_load_time_ms
    - checkout_abandonment
    - refund_requests
  alpha: 0.01

stage2:
  contextual_columns:
    - release_flag
  rolling_window: 7

stage3:
  hypotheses:
    H_release_defect:
      description: "Release introduced a functional or performance defect"
      prior: 0.35
    H_infra_scaling:
      description: "Infrastructure capacity or scaling issue"
      prior: 0.20
    H_user_learning:
      description: "Temporary user learning curve"
      prior: 0.20
    H_external_shift:
      description: "External traffic quality change"
      prior: 0.15
    H_data_lag:
      description: "Analytics or logging delay"
      prior: 0.10

stage4:
  narrative:
    max_sentences: 3
    tone: "executive"

stage5:
  thresholds:
    high_confidence: 0.75
    medium_confidence: 0.50
