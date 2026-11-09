# Entangle

An honest quantum machine-learning benchmark on data that is quantum to begin
with: quantum states, not text or tables squeezed into rotation angles.

Work in progress, built in stages (see branches and pull requests).

## Stage 1: entanglement classifier

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python stage1_entanglement.py        # 2, 3, 4 qubits, about 15 seconds
.venv/bin/python -m pytest -q tests
```

Results: [results/stage1_entanglement.md](results/stage1_entanglement.md)
