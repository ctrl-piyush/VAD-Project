# Video Anomaly Detection using CNN–Transformer (UCF-Crime, weakly supervised MIL)

Pipeline: 16-frame snippets → frozen pretrained 3D-CNN features (cached) → Transformer temporal encoder → MIL ranking
loss (video labels only) → snippet scores → localisation → calibrated confidence → (optional) RAG explanation + groundedness
check → Streamlit dashboard.

## Start here
**Using the Kaggle frame dataset (odins0n/ucf-crime-dataset)? Read KAGGLE_GUIDE.md first** (it is images, not videos).

Read **GUIDE_STEPS.md** (how to train/test, Hinglish) and **SIR_BRIEFING.md** (what to tell your guide). Whole study in one command:
`python scripts/run_all.py --seeds 1 2 3 --with-explanations`, then `python scripts/make_report.py`.

## What has actually been run (honest status)
Verified in a CPU sandbox on **synthetic** data: `pytest -q` (unit tests for metrics/calibration/stats/masks/split/CLIP-grid/explanation checker,
an end-to-end train→resume→evaluate test), the full `scripts/run_all.py` (validate → split → features → analysis → tune → 3 groups × seeds →
evaluate → report) including the CLIP-ablation code path (with stand-in features), the FastAPI endpoints (TestClient), all 8 dashboard pages
(Streamlit AppTest), offline W&B logging. **Not verified (needs your machine):** real UCF-Crime files/annotation format, Kinetics weight
download, CUDA/fp16, real CLIP/NLI/BLIP-2/FAISS model loading (pure logic is tested), real extraction speed. No UCF-Crime result exists yet.
Optional API/app tests: `VAD_TEST_FIXTURE_RUN=<run> VAD_TEST_VIDEO=<mp4> pytest tests/test_api_app.py` (needs a trained run).

## Setup
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# install torch/torchvision from https://pytorch.org for your CUDA, then:
pip install -r requirements.txt
python scripts/check_hardware.py                          # send me the JSON if something looks off
pytest -q                                                 # should be green before you touch real data
```

## Dataset (you do this manually)
1. Get UCF-Crime from the official source (Sultani et al., project page / CRCV UCF) and its annotation files.
2. Put files like this (subfolders under `videos/` don't matter):
```
data/ucf_crime/videos/...mp4
data/ucf_crime/annotations/Anomaly_Train.txt
data/ucf_crime/annotations/Anomaly_Test.txt
data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```
   Other location: `export VAD_DATA_ROOT=/path/ucf_crime` and `export VAD_ARTIFACTS=/path/artifacts`.
   Open the temporal file once: the parser expects `name class s1 e1 s2 e2` (frame numbers, `-1` = absent). If yours differs, edit `parse_temporal` in `vad/data/validate.py`.

## Run order
```
python -m vad.data.validate --limit 20          # quick check, then without --limit (add --hash for duplicate content)
python -m vad.data.validate                     # must exit 0; read artifacts/dataset/validation_report.json
python -m vad.data.make_val_split               # fixed, stratified 10% of TRAIN -> val (test untouched)
python -m vad.data.features --limit 10          # smoke test; note seconds_per_new_video in artifacts/reports/extraction_report.json
python -m vad.data.features                     # full run, resumable (re-run after any interruption)
python -m vad.analysis [--umap]                 # imbalance, lengths, MMD / Frechet shift
python -m vad.train --run-name base_s42 --seed 42 --set model.type=baseline     # baseline first
python -m vad.train --run-name tf_s42 --seed 42                                  # hybrid
python -m vad.tune                              # validation-driven table (configs/tuning.yaml), one seed each
python scripts/run_final.py --name tf   --seeds 1 2 3 [--set <best overrides>]
python scripts/run_final.py --name base --seeds 1 2 3 --set model.type=baseline
python -m vad.evaluate --group tf --runs tf_s1 tf_s2 tf_s3 --vs base=base_s1,base_s2,base_s3 cliptf=cliptf_s1,cliptf_s2,cliptf_s3   # TEST, once
streamlit run app/streamlit_app.py
```
`--set key=value` overrides any config entry (e.g. `--set train.lr=0.0003 model.n_layers=1`).

## Protocol notes (read these)
- **Frame-level ROC-AUC** (concatenated test frames, needs temporal annotations) is the number comparable to the literature.
  **Video-level AUC** (max snippet score vs. label) is reported separately and is not comparable with it.
- Validation has only video labels, so checkpoint/hyperparameter selection uses **val video AUC**. The operating threshold is
  F1-optimal on val. Localisation parameters (`eval.*`) cannot be tuned against frame labels without touching test; they are fixed
  config values, and temporal IoU is reported once on test. Papers that pick the best epoch on test AUC will look better than this protocol.
- Normalisation (`zscore`) is fit on train bags only; the retrieval store asserts every indexed video is in the train split.
- One-seed tuning differences of ~0.01 AUC are noise; trust the multi-seed mean ± SD and paired bootstrap CIs, not single runs.
- Calibration: the MIL score is not a probability. A temperature (optional bias) is fit on val; ECE/Brier are reported on test.
  With ~160 val videos the fit is noisy. Seed spread is used for routing to human review.
- Groundedness threshold in `configs/default.yaml` is a placeholder until tuned on ~150 human-labelled explanations
  (`vad/explain/groundedness.py: tune_threshold / checker_report`). For retrieval-only text the NLI check is circular (flagged in output).
- Expected tuning levers (in order of usual payoff): feature quality (backbone/weights, I3D if you obtain verified weights, multi-crop),
  `model.segments`, lr/weight decay/dropout, `model.n_layers`, class balancing, smoothness/sparsity weights. Nothing here promises a particular AUC.

## Explanations (optional, after the detector works)
```
pip install transformers accelerate pillow faiss-cpu      # faiss optional (numpy fallback)
python -m vad.infer fit --run tf_s1
python -m vad.explain.build_index --run tf_s1 [--captions-json your_UCA_captions.json]
```
then use the dashboard "Explanations" page. BLIP-2 is a multi-GB download; leave it unchecked for retrieval-only text.

## Layout
`vad/` library (data/validate, make_val_split, features, bags; models, train, evaluate, metrics, localize, calibration, stats, analysis,
infer, tune, explain/*), `app/` dashboard, `configs/`, `scripts/`, `tests/`.
