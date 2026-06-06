# CS2 Match Predictor

Predict the winner of a Counter-Strike 2 match from the 10 players' FACEIT
usernames. Enter two teams of five, and the app estimates each player's skill
from their recent match history and returns a calibrated win probability.

The project is built end-to-end: a **leakage-safe ML pipeline** (FACEIT API →
features → model) and a **Django web app** with a caching layer and
persistence, both sharing the same prediction core.

---

## Why this project is interesting

Most "predict the winner" projects quietly leak future information and report
implausibly high accuracy. This one is deliberately careful about that, and is
honest about the result:

- **Leakage-safe by construction.** A match's features are built *only* from
  each player's matches that finished **before** that match. The guard is
  verified, not assumed (see `src/faceit_api.py::features_before`).
- **Chronological train/test split** — the model is always tested on matches
  newer than everything it trained on.
- **Measured against a baseline mindset** — the goal is beating a coin flip on
  *balanced* matchmaking, the hardest possible case.
- **Honest result:** **AUC ≈ 0.60** on 243 real matches. A believable, modest
  edge — and the fact that it *isn't* 0.85+ is itself evidence the leakage
  protection works.

## Architecture

```
                ┌─────────────────────────────────────────┐
                │  src/  (framework-agnostic ML library)   │
                │  faceit_api · features · dataset · train │
                │  predict                                 │
                └───────────────┬─────────────────────────┘
                                │ predict_from_stats()
                  ┌─────────────┴─────────────┐
                  │                           │
          ┌───────▼──────┐            ┌───────▼───────┐
          │  Django app  │            │  CLI (python  │
          │ (config/ +   │            │  -m src.*)    │
          │  predictor/) │            └───────────────┘
          │ cache · DB · │
          │ tests        │
          └──────────────┘
```

The ML logic lives in `src/` and knows nothing about the web. The Django app is
a thin layer over it that adds production concerns.

### Scaling decision: caching the slow external API

A single prediction makes **10 calls** to the rate-limited FACEIT API (~5–6s
cold). The Django service caches each player's stats, so repeat predictions are
served from memory:

| | Cold (live FACEIT) | Warm (cached) |
|---|---|---|
| Time for a prediction | ~5.6 s | ~0.05 s (**~100× faster**) |

In production this in-memory cache would move to Redis, and the FACEIT fetches
into a background worker so requests never block.

## Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic regression (baseline) | 0.531 | **0.596** |
| XGBoost | 0.510 | 0.531 |

On 243 leakage-safe matches (chronological split, 49 in test). The numbers are
noisy at this sample size — treated as promising signal, not proof. The model is
far more confident on *unbalanced* matchups (e.g. a clearly stronger team scores
~0.88), which is the intended use case.

## Project structure

```
src/                    # ML library (no web dependencies)
  faceit_api.py         #   FACEIT client + leakage-safe stat aggregation
  features.py           #   team-difference feature vectors
  dataset.py            #   builds labeled training data from real matches
  train.py              #   chronological split, trains + evaluates, saves model
  predict.py            #   matchup → win probability
config/                 # Django project (settings, urls)
predictor/              # Django app: views, forms, services (cache+model), model, tests
data/processed/         # demo dataset + trained model (committed for reproducibility)
```

## Getting started

```bash
# 1. Setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. FACEIT API key (free, from developers.faceit.com)
cp .env.example .env        # then set FACEIT_API_KEY=...

# 3. Run the Django web app
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver      # http://localhost:8000

# 4. Or use the CLI
.venv/bin/python -m src.predict \
  --team-a='name1,name2,name3,name4,name5' \
  --team-b='name6,name7,name8,name9,name10'

# 5. Rebuild the dataset / retrain (optional)
.venv/bin/python -m src.dataset --seed='<nickname>' --max-matches 400
.venv/bin/python -m src.train
```

Run the tests:

```bash
.venv/bin/python manage.py test predictor
```

## Limitations & future work

- **More data.** 243 matches is small; thousands would firm up the AUC estimate.
- **Elo baseline.** A reconstructed (leakage-safe) Elo rating would be both a
  stronger baseline to beat and a likely-powerful feature.
- **Richer features.** The FACEIT stats endpoint also exposes K/R, MVPs,
  per-map performance, etc.
- **Deployment.** Add gunicorn/whitenoise and host on a free tier for a live link.

## Tech stack

Python · scikit-learn · XGBoost · pandas · Django · FACEIT Data API
