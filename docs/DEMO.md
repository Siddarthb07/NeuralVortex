# NeuralVortex demo (screenshots)

Assets are generated locally (they are **not** bundled from CI — regenerate after changing the solver or training loop):

```bash
pip install -r requirements.txt matplotlib
pip install -r requirements-train.txt
python scripts/capture_demo_assets.py            # plots + short train + eval
pip install -r requirements-demo.txt
python demo/app_gradio.py                        # Gradio on http://127.0.0.1:7860
```

## Static captures (matplotlib)

| File | Description |
|------|-------------|
| [`assets/demo/velocity_slice.png`](assets/demo/velocity_slice.png) | Velocity magnitude slice from `data/smoke.h5`. |
| [`assets/demo/training_loss.png`](assets/demo/training_loss.png) | Train/val MSE during a short Conv3D smoke run. |
| [`assets/demo/pooled_mae.png`](assets/demo/pooled_mae.png) | Pooled MAE vs solver scalars on the validation shard. |
| [`assets/demo/surrogate_inference_panel.png`](assets/demo/surrogate_inference_panel.png) | Pooled outputs for fixed slider settings (same mapping as `demo/app_gradio.py`). |

## Live Gradio UI

```bash
pip install -r requirements-demo.txt   # pins huggingface_hub for Python 3.9 + Gradio 4.x
python demo/app_gradio.py              # http://127.0.0.1:7860
```

Use your browser’s screenshot tool if you want a pixel-perfect UI clip; the inference panel PNG above is generated from the **same** checkpoint and slider values for reproducible docs.
