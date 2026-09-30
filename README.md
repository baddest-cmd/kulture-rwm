# `kulture-rwm`

> **Model-Based Reinforcement Learning for Cultural Alignment in Recommender Systems**

An industrialised **World Model** framework designed to eliminate popularity magnitude bias and algorithmic flattening in cultural music recommendation.

---

## Evolution
* **From `for-the-kulture`:** Upgrades static offline representation alignment into a closed-loop **Gymnasium environment** (`TidalKultureGymEnv`) to simulate multi-session listener fatigue and Context Tax ($\tau_c$) dynamics.
* **From TIDAL Daily Discovery:** Retains TIDAL's SASRec sequence backbone for candidate retrieval, replacing myopic heuristic filters with an **Action-Conditioned RSSM** dynamics engine and a **CEM-MPC Trajectory Planner**.

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/baddest-cmd/kulture-rwm.git
cd kulture-rwm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Code Quality & Unit Tests
```bash
ruff check .
pytest -v
```

### 3. Containerised Execution (Docker)
```bash
docker build -t kulture-rwm:latest .
docker run --rm kulture-rwm:latest
```

### 4. Training & Benchmarking
```bash
# Train Recurrent State Space Model (50 Epochs)
python train_world_model.py --epochs 50 --use_wandb

# Run 30-Session Closed-Loop Benchmark
python run_simulation_benchmark.py --use_wandb
```

---

## Benchmark Metrics

| Evaluation Metric | TIDAL Baseline | kulture-rwm |
| :--- | :--- | :--- |
| Catalogue Exposure Gini ($\mathcal{L}_{\text{gini}}$) | $0.9605$ | $0.9437$ |
| Subgenre Shannon Diversity ($\mathcal{H}$) | $2.799 \text{ bits}$ | $2.802 \text{ bits}$ |
| Context Tax Accumulation ($\tau_c$) | $0.00$ | $0.00$ |

---

## License
Apache 2.0
