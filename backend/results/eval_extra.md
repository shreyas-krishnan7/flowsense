# Evaluation gaps — selected model (Transformer + CE, seed 1)

## Calibration (known test flows)
- Softmax head ECE: 0.0043
- k-NN vote-share ECE: 0.0133
- Reliability diagram: runs/transformer_ce_p30_s1/plots/reliability.png

## Open-set (20 known services + unknown), novelty threshold from validation (95% known retention)
- Open-set macro-F1 (21 classes): 83.61%
- Unknown-class F1: 78.63%
- Unknown recall: 81.2% | known false-alarm on test: 8.4% (5% intended on validation -> threshold drifts across weeks)
- Accuracy on accepted known flows: 93.42% | closed-set macro-F1 without rejection: 88.85%

## Unknown detection by provider
| Provider | Detected | Flows |
|---|---|---|
| google | 85.6% | 39,883 |
| (independent) | 68.9% | 6,836 |
| (unmapped) | 52.1% | 2,889 |
| alza | 52.5% | 356 |
| overleaf | 61.1% | 36 |

## Unknown detection by service category
| Category | Detected | Flows |
|---|---|---|
| Streaming media | 84.6% | 13,153 |
| Other services and APIs | 74.5% | 12,139 |
| Advertising | 88.4% | 5,538 |
| Search | 89.8% | 5,105 |
| default | 89.0% | 3,309 |
| (unmapped) | 52.1% | 2,889 |
| Antivirus | 96.2% | 2,373 |
| Authentication services | 93.7% | 1,786 |
| File sharing | 97.1% | 1,067 |
| E-commerce | 47.9% | 631 |
| Analytics & Telemetry | 52.7% | 609 |
| Games | 64.4% | 570 |
| Blogs & News | 46.1% | 345 |
| Information systems | 71.1% | 225 |
| Social | 64.5% | 186 |
| Videoconferencing | 91.4% | 35 |
| Mail | 13.8% | 29 |
| Music | 100.0% | 11 |

## Source-IP separation (known test flows)
- 21.7% of known test flows come from source IPs never seen in training (32,566 flows)
- Accuracy: unseen IPs 92.27% vs seen IPs 93.01% (macro-F1 unseen 89.09%)
- Conclusion: no evidence the model relies on memorising client hosts.
