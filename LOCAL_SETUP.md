# Run NeuralVortex locally

```powershell
cd NeuralVortex-local
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-baseline.txt
python data/generate.py --n 8 --grid-res 16 --out data/smoke.h5
python train/train_sklearn_baseline.py --h5 data/smoke.h5
pytest tests/
```

Full FNO training: see `train/README.md` and `requirements-train.txt` (GPU recommended).
