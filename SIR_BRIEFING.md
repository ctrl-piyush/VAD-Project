# What to tell your guide (Dr. Sanjay Roka)

## 1. One-minute summary (say this first)
"Sir, we have implemented the full pipeline from our synopsis as one repository: pretrained 3D-CNN features cached to disk, a Transformer
temporal encoder, MIL ranking loss on video-level labels, temporal localisation, calibrated confidence with human-review routing,
a retrieval-augmented explanation module with a groundedness check, and a Streamlit dashboard with a FastAPI backend.
Code is tested on synthetic data. We are now running it on UCF-Crime."

## 2. What is DONE (you can show the code / dashboard)
- Dataset validator (missing/corrupt/duplicate videos, train-test leakage check), fixed stratified validation split
- Resumable feature extraction (ResNet3D-18 Kinetics; CLIP-ViT for the ablation)
- Models: CNN+MIL baseline, Transformer+MIL (CLIP features), CNN–Transformer+MIL (proposed); masks for variable length
- Metrics: frame-level and video-level ROC-AUC (kept separate), AP, F1, false-alarm rate, per-category AUC, temporal IoU
- Statistics: 3 seeds mean ± SD, paired bootstrap 95% CI, Holm–Bonferroni
- Calibration: temperature scaling, ECE, Brier, risk–coverage, human-review routing
- Dataset analysis: class imbalance, length distribution, MMD / Fréchet distance (optionally UMAP)
- Explanation: FAISS-style retrieval from TRAIN videos only, BLIP-2, CLIP-score + NLI groundedness, labelling workflow with Cohen's kappa
- Dashboard (8 pages) + FastAPI + optional Weights & Biases logging; `scripts/run_all.py` reproduces everything with one command
- 21 automated tests pass (+2 optional API/app tests); all pages of the dashboard load without errors

## 3. Be honest about these (sir will respect it)
- Real UCF-Crime results do not exist until the run finishes; the numbers we have are from a synthetic smoke test only.
- We evaluate with a strict protocol: the checkpoint and threshold are chosen on a validation split, never on the test set. Published papers often select the best epoch on test AUC, so our numbers may be a bit lower but they are defensible.
- The hybrid model is not guaranteed to beat the baseline; the ablation will tell us, and we will report whatever it shows.
- I3D is not included: the synopsis says "I3D / ResNet3D", and we use ResNet3D-18 (torchvision, Kinetics-400).
- The explanation module (BLIP-2) is the "Extended" tier; its human labelling needs two annotators (~150 explanations).

## 3b. If we use the Kaggle frame dataset instead of the original videos (tell sir this clearly)
The Kaggle copy (odins0n/ucf-crime-dataset) contains every 10th frame of the official UCF-Crime videos as 64x64 PNG images, not videos.
Consequences: a "16-frame snippet" covers ~5 s instead of ~0.5 s, motion cues are weaker, localisation is coarser, results are NOT comparable with
published UCF-Crime numbers, and temporal annotations must be obtained separately (otherwise only video-level AUC). Ask sir whether this is acceptable
as a documented fallback, or whether we must use the original videos. The code supports both (`configs/kaggle_frames.yaml` vs `configs/default.yaml`).

## 4. When to meet sir (milestones, not dates)
- **Meeting 1: after the first real results.** Show: validation report, dataset analysis (imbalance, lengths, feature shift), baseline val AUC, the dashboard running. Ask: is ResNet3D-18 acceptable instead of I3D; is the strict validation protocol OK; who can be the 2nd annotator.
- **Meeting 2: after final 3-seed evaluation.** Show: ablation table with confidence intervals, per-category AUC, calibration plots, localisation examples. Ask for feedback on the Discussion/Limitations.
- **Meeting 3: before submission.** Show the explanation labelling results, the full `report.docx`, and a live dashboard demo.
If something blocks you (GPU, dataset access), tell sir early instead of waiting for the next milestone.

## 5. Live demo order (5 minutes)
1. Dataset page: validation report + class imbalance  2. Training page: logs/curves/checkpoints  3. Analyze video: upload a clip → score timeline, predicted intervals, calibrated confidence, routing
4. Evaluation + Trends: AUC with CI, per-category bars, false-alarm sweep  5. Explanations page: retrieved evidence, explanation, groundedness (say: "consistency evidence, not proof")

## 6. Likely questions
- *Why MIL?* Frame labels are expensive; only video labels are needed.  *Why a Transformer?* Long-range temporal context; the ablation measures it.
- *Why not accuracy?* Anomalies are rare; ROC-AUC, false-alarm rate and calibration matter more.  *How do you avoid leakage?* Fixed val split from train, normalisation fit on train only, retrieval store asserted train-only, test used once.
