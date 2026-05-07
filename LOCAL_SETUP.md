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
pip install -r requirements-train.txt
python train/train_tfno.py --no-tfno --epochs 25 --cpu-only --h5 data/smoke.h5
python train/eval_surrogate.py --no-tfno --cpu-only
python train/pinn_residual_smoke.py --h5 data/smoke.h5
pip install -r requirements-demo.txt
# python demo/app_gradio.py
pytest tests/
```

Full TFNO / GPU + W&B: see `train/README.md` and `requirements-train.txt`.

Gradio UI is optional (`demo/app_gradio.py`).
