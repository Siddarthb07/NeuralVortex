# NeuralVortex demos (Phase 4)

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
pip install -r requirements-demo.txt
python train/train_tfno.py --no-tfno --epochs 40 --cpu-only --h5 data/smoke.h5
python demo/app_gradio.py
```

The UI reads pooled velocity/pressure scalars from `runs/tfno_train/surrogate_best.pt`. Train with `--no-tfno` on laptops without a stable `neuralop` wheel.

## Hugging Face Space (outline)

1. Create a **Gradio** Space (Python 3.10+).
2. Copy `demo/app_gradio.py`, `train/` helpers (`infer_surrogate.py`, `field_dataset.py`, `surrogate_models.py`, `hdf5_loader.py`), and a **small** checkpoint.
3. Pin deps in Space `requirements.txt`: `torch`, `gradio`, `numpy`, `h5py`, and optionally `neuraloperator`.
4. Entrypoint: `python app_gradio.py` after flattening paths **or** set `PYTHONPATH` to repo root.

CPU Spaces should use Conv3D checkpoints (`--no-tfno`) and small grids only.
