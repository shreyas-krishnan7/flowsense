# E7: few-shot registration of unseen services — negative result

Method: prototype (nearest class-mean) classification. Known services use training-embedding means; each new service uses the mean of K labeled flows. New services: 10 unseen CESNET services, at most one per provider: google-www, google-background, default-background, fontawesome, joinhoney, playradio, sme-sk, onesignal, alza-www, tiktok. 5 random draws per K.

Pre-registered criterion: at K=10, new-service accuracy ≥ 70% and known-service accuracy drop ≤ 2 points.

| Embedding | New acc K=1 | K=5 | K=10 | K=20 | k-NN new acc K=10 | Known acc 0-shot → K=10 | Criterion |
|---|---|---|---|---|---|---|---|
| Transformer + CE (selected) | 21.5 | 32.6 | 34.6 | 36.1 | 0.9 | 89.8 → 85.0 | FAIL |
| Transformer + SupCon | 14.0 | 13.4 | 13.8 | 15.4 | 0.1 | 90.9 → 88.3 | FAIL |
| TCN + SupCon | 21.6 | 24.9 | 26.6 | 27.0 | 0.4 | 89.8 → 86.7 | FAIL |

**Decision:** few-shot registration is **not supported** by these embeddings and is removed from the project claims. Embeddings trained to separate 20 known services do not separate unseen services well; contrastive (SupCon) embeddings do not help. Adding new services requires retraining (or future work: episodic/meta-learning objectives).
