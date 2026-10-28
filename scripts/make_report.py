"""
Builds report/report.md (+ report.docx if pandoc exists) from YOUR real results in artifacts/reports/.
Prose sections are written for you (from the synopsis); every number comes from the JSON files; missing results show
"[NOT RUN YET]". Nothing is invented. Re-run any time:   python scripts/make_report.py
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import pandas as pd
from vad.config import load_config, paths

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--group", default="tf")
a = ap.parse_args()
cfg = load_config(a.config); P = paths(cfg)
NA = "[NOT RUN YET]"


def J(p):
    p = Path(p); return json.load(open(p)) if p.exists() else None


def f(x, d=4):
    return NA if x is None or x != x else f"{x:.{d}f}"


def ms(d):
    return NA if not d else (f"{d['mean']:.4f} ± {d['sd']:.4f}" if d.get("sd") is not None else f"{d['mean']:.4f}")


def ci(c):
    return "n/a" if not c else f"[{c['lo']:.4f}, {c['hi']:.4f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


ev = J(P["reports"] / f"eval_{a.group}.json"); da = J(P["reports"] / "dataset_analysis.json")
vr = J(P["dataset"] / "validation_report.json"); xe = J(P["reports"] / "explanation_eval.json"); tun = P["reports"] / "tuning_table.csv"
choice = J(P["reports"] / "final_config_choice.json")
smoke = False
for c in P["runs"].glob("*/config.json") if P["runs"].exists() else []:
    if J(c)["cfg"]["preprocess"].get("weights") == "none": smoke = True
names = {"tf": "CNN–Transformer + MIL (proposed)", "base": "CNN + MIL (baseline)", "cliptf": "Transformer + MIL (CLIP ViT features)"}

L = []
A = L.append
A("# Video Anomaly Detection using CNN–Transformer: Weakly-Supervised MIL with Calibrated, Evidence-Grounded Explanations\n")
A("**Authors:** Lakshay Verma, Piyush, Khushi Sharma  |  **Guide:** Dr. Sanjay Roka  |  Team ID MP2026CSE162  |  Graphic Era Deemed to be University, Dehradun\n")
if smoke:
    A("> **WARNING: these numbers come from a SYNTHETIC / random-weights smoke test and are NOT results on UCF-Crime. Do not submit them.**\n")
A("## Abstract\n")
if ev:
    e = ev["ensemble"]; b = ev["bootstrap_95ci"]
    A(f"We study weakly-supervised video anomaly detection on UCF-Crime with a frozen 3D-CNN backbone, a Transformer temporal encoder and a Multiple Instance Learning ranking loss. "
      f"Over {len(ev['runs'])} seeds the proposed model reaches a frame-level ROC-AUC of {ms(ev['mean_sd_across_seeds']['frame_auc'])} (mean ± SD) and a video-level ROC-AUC of {ms(ev['mean_sd_across_seeds']['video_auc'])}; "
      f"the seed-ensemble frame-level AUC is {f(e['frame_auc'])} (95% paired-bootstrap CI {ci(b.get('frame_auc'))}). We additionally report temporal localisation, calibration (ECE/Brier), ablations with significance tests and an evidence-groundedness check for generated explanations.\n")
else:
    A(f"{NA} (run the evaluation stage).\n")

A("## 1. Introduction\n")
A("Surveillance infrastructure has grown faster than the human capacity to watch it: a control room may receive hundreds of feeds while anomalous events are rare, unpredictable and last only seconds. Automated video anomaly detection (VAD) scores incoming video and raises an alert only when something unusual happens. "
  "Early VAD relied on hand-crafted motion features or single-scene autoencoders that generalise poorly; 3D CNNs improved local motion representation but struggle to relate events separated by long intervals, which motivates pairing a CNN feature extractor with a Transformer encoder [1]. "
  "Frame-level annotation is costly, so anomaly detection is commonly framed as Multiple Instance Learning (MIL) with video-level labels only [2,3]. Vision-language models such as CLIP [4] and BLIP-2 [5] can describe a flagged event, but a fluent description may be ungrounded in the footage, a risk in security settings. "
  "Finally, benchmarks are imbalanced and usually compared with single-run AUC point estimates while scores are not calibrated probabilities. This work therefore combines (i) hybrid CNN–Transformer temporal modelling, (ii) MIL weak supervision, (iii) evidence-grounded explanations, and (iv) statistically rigorous, calibrated evaluation.\n")
A("## 2. Related Work\n")
A("Sultani et al. [2] introduced MIL ranking for real-world surveillance anomaly detection and released UCF-Crime. Tian et al. [3] proposed Robust Temporal Feature Magnitude learning, a common baseline. Vision-language approaches include VadCLIP [8] (frozen CLIP with dual branches; 88.02% AUC on UCF-Crime in its paper) and the training-free LAVAD [6]. "
  "For explanations, Lv and Sun [9] use large language models and SlowFastVAD [7] combines a light detector with a retrieval-augmented VLM. Four gaps remain: few systems verify that an explanation is supported by visual evidence; latency is rarely discussed; results are single-run point estimates without confidence intervals; and calibration is rarely checked [13]. "
  "We treat explanation faithfulness as measurable, report uncertainty around headline metrics, and analyse dataset shift and calibration.\n")
A("## 3. Method\n")
A("**Features.** Videos are split into 16-frame snippets; each snippet is embedded by a frozen, pretrained 3D CNN (ResNet3D-18 pretrained on Kinetics-400) and cached to disk once, so all training and ablation runs use compact feature tensors. "
  "**Temporal model.** Snippet features are resampled to a fixed number of segments, projected, given learned positional embeddings and processed by a pre-norm Transformer encoder with multi-head self-attention; an MLP head outputs a score per segment (padding masks are supported for variable-length input). "
  "**MIL objective.** Following [2], the highest-scoring segment of an anomalous video is pushed above that of a normal video with a hinge ranking loss, plus temporal-smoothness and sparsity terms on anomalous videos; batches contain equal numbers of anomalous and normal videos and anomaly classes are sampled uniformly to counter class imbalance. "
  "**Localisation.** Segment scores are mapped back to snippets, smoothed and thresholded relative to the video maximum to obtain start/end times. "
  "**Calibration and routing.** MIL scores are not probabilities; temperature scaling is fitted on a validation split, ECE/Brier/risk–coverage are reported on test, and borderline or seed-inconsistent cases are routed to human review. "
  "**Explanations.** Flagged snippets retrieve similar past incidents (CLIP embeddings; store built from training videos only), BLIP-2 generates a one-sentence explanation, and a groundedness score = min(CLIP-score percentile, NLI entailment) decides whether it is accepted.\n")

A("## 4. Experimental Setup\n")
if vr:
    s = vr["stats"]
    A(f"UCF-Crime (13 anomaly categories + normal). Train list: {s['train']['n_listed']} videos; test list: {s['test']['n_listed']} videos. "
      f"A fixed, class-stratified {int(cfg['data']['val_fraction']*100)}% of the training videos (seed {cfg['data']['val_seed']}) forms the validation split; the official test split is used only for final reporting.\n")
else:
    A(f"Dataset statistics: {NA}\n")
cc = P["runs"] / f"{a.group}_s{ev['runs'][0].split('_s')[-1]}" / "config.json" if ev else None
cj = J(cc) if cc else None
if cj:
    t, m = cj["cfg"]["train"], cj["cfg"]["model"]
    A(f"Hyper-parameters (proposed model): d_model={m['d_model']}, layers={m['n_layers']}, heads={m['n_heads']}, dropout={m['dropout']}, segments T={m['segments']}, feature norm={m['feature_norm']}; "
      f"AdamW lr={t['lr']}, weight decay={t['weight_decay']}, batch {t['batch_size']}, {t['epochs']} epochs max with early stopping on validation video-AUC (patience {t['patience']}). "
      f"Device: {cj['device']}, PyTorch {cj['torch']}. Seeds: {', '.join(r.split('_s')[-1] for r in ev['runs'])}.\n")
if choice: A(f"Hyper-parameter selection: validation-driven, single seed per candidate; chosen configuration: {choice['chosen']}.\n")
if tun.exists():
    A("**Validation tuning table** (video-level AUC on the validation split only):\n"); A(pd.read_csv(tun).to_markdown(index=False) + "\n")
A("**Evaluation protocol.** *Frame-level ROC-AUC* is computed over all frames of the test videos with temporal annotations (normal test videos contribute only negatives); it is the figure comparable with the literature. "
  "*Video-level AUC/AP* use the maximum segment score per video and are reported separately. The operating threshold is the F1-optimal value on the validation split. "
  "Mean ± SD is over seeds; confidence intervals are 95% paired bootstrap over test videos (≥1,000 resamples) of the seed-ensemble; multiple comparisons use Holm–Bonferroni.\n")

A("## 5. Results\n")
if ev:
    A("### 5.1 Detection and ablation (detectors)\n")
    rows = [[names["tf"], ms(ev["mean_sd_across_seeds"]["frame_auc"]), ms(ev["mean_sd_across_seeds"]["video_auc"]), f(ev["ensemble"]["frame_auc"]), ci(ev["bootstrap_95ci"].get("frame_auc")), f(ev["ensemble"]["video_auc"]), ci(ev["bootstrap_95ci"]["video_auc"])]]
    for k, o in (ev.get("other_models") or {}).items():
        rows.append([names.get(k, k), ms(o["mean_sd_across_seeds"]["frame_auc"]), ms(o["mean_sd_across_seeds"]["video_auc"]), f(o["ensemble"]["frame_auc"]), ci(o["bootstrap_95ci"].get("frame_auc")), f(o["ensemble"]["video_auc"]), ci(o["bootstrap_95ci"]["video_auc"])])
    A(table(["Model", "Frame AUC (mean±SD)", "Video AUC (mean±SD)", "Frame AUC ens.", "95% CI", "Video AUC ens.", "95% CI"], rows) + "\n")
    if ev.get("comparisons"):
        A("**Paired comparisons** (proposed minus other; significant only if the 95% CI excludes zero after Holm correction):\n")
        A(table(["Comparison", "Δ", "95% CI", "Holm-adj. p", "Significant"], [[k, f(c["diff"]), f"[{c['lo']:.4f}, {c['hi']:.4f}]", f(c["holm_adjusted_p"]), "yes" if c["significant_after_holm"] and c["excludes_zero"] else "no"] for k, c in ev["comparisons"].items()]) + "\n")
        for k, c in ev["comparisons"].items():
            A(f"- `{k}`: difference {c['diff']:+.4f}, CI {'excludes' if c['excludes_zero'] else 'includes'} zero.")
        A("")
    A("### 5.2 Per-category performance\n")
    A(table(["Category", "# test videos", "Video AUC", "Frame AUC"], [[c, v["n_videos"], f(v["video_auc"]), f(v.get("frame_auc"))] for c, v in ev["per_category"].items()]) + "\n")
    op, ti = ev["operating_point"], ev["temporal_iou"]
    A("### 5.3 Operating point and localisation\n")
    A(f"Threshold {op['thr']:.3f} (chosen on validation): precision {op['precision']:.3f}, recall {op['recall']:.3f}, F1 {op['f1']:.3f}, video-level false-positive rate {op['fpr']:.3f}; "
      f"false-alarm intervals per hour of normal video: {f(op['false_alarm_intervals_per_normal_hour'], 1)}. Mean temporal IoU on {ti['n']} annotated anomalous test videos: {f(ti['mean'])} (localisation parameters are fixed config values, not tuned on test).\n")
    cal = ev["calibration"]
    A("### 5.4 Calibration\n")
    A(table(["Scores", "ECE", "Brier"], [["raw MIL score (not a probability)", f(cal["raw"]["ece"]), f(cal["raw"]["brier"])], ["temperature scaling (val-fit)", f(cal["temperature"]["ece"]), f(cal["temperature"]["brier"])], ["temperature + bias", f(cal["temperature_bias"]["ece"]), f(cal["temperature_bias"]["brier"])]]) + "\n")
    A(f"Routing of test videos: {ev['routing_counts']}.\n")
    png = P["reports"] / f"eval_{a.group}_calibration.png"
    if png.exists(): A(f"![Reliability diagram and risk–coverage curve]({png})\n")
else:
    A(f"{NA} Run `python -m vad.evaluate` (or `scripts/run_all.py`).\n")
A("### 5.5 Explanation groundedness\n")
if xe and isinstance(xe.get("checker"), dict):
    A(f"{xe['n_labelled']} explanations labelled by two annotators (Cohen's κ = {xe['cohen_kappa']:.3f}; {xe['n_disagreements_dropped']} disagreements dropped).\n")
    A(table(["Variant", "Hallucination rate", "n"], [[v, f(d["rate"]), d["n"]] for v, d in xe["hallucination_rate_by_variant"].items()]) + "\n")
    A(table(["Checker", "Precision (ungrounded)", "Recall (ungrounded)", "ROC-AUC", "AUC 95% CI"], [[k, f(v["precision_ungrounded"]), f(v["recall_ungrounded"]), f(v.get("roc_auc")), ci(v.get("roc_auc_ci"))] for k, v in xe["checker"].items()]) + "\n")
    g = xe["with_evidence_grounding"]; A(f"With evidence-grounding (threshold {g['threshold']:.3f}): acceptance rate {f(g['acceptance_rate'])}; hallucination rate {f(g['hallucination_rate_before'])} → {f(g['hallucination_rate_after'])}.\n")
else:
    A(f"{NA} (build the retrieval store, label the generated explanations, run `python -m vad.explain.evaluate_checker`). This component is the 'Extended' tier.\n")
A("### 5.6 Dataset analysis\n")
if da:
    imb = da["imbalance"].get("train", {})
    A(f"Training anomaly classes: imbalance ratio (max/min) {imb.get('imbalance_ratio')}, entropy {imb.get('entropy_bits')} of {imb.get('max_entropy_bits')} bits.\n")
    for k in ("shift_train_vs_test", "shift_normal_vs_anomalous_train"):
        if k in da: A(f"- {k}: MMD² = {da[k]['mmd2']:.5f}, Fréchet distance = {da[k]['frechet']:.3f}")
    ov = P["reports"] / "dataset_overview.png"
    if ov.exists(): A(f"\n![Dataset overview]({ov})\n")
else:
    A(f"{NA}\n")

A("## 6. Discussion\n")
A("_Write 2–3 paragraphs in your own words, using only the tables above: (a) did the Transformer help over the CNN+MIL baseline, and is the difference significant after Holm correction? (b) which categories are easy/hard and how does this relate to class imbalance and feature shift? (c) how well calibrated is the score and what does the risk–coverage curve say about human review? (d) what did the groundedness check catch? The auto-generated sentences in 5.1 state the statistical facts you can build on._\n")
A("## 7. Limitations\n")
A("Features come from a frozen backbone, so fine-tuning is not explored. Validation has only video-level labels, therefore localisation parameters cannot be tuned against frame labels without touching the test set; many published numbers select the best epoch on test AUC and are not directly comparable to this stricter protocol. "
  "Calibration is fitted on a small validation set. The groundedness check is validated on a small human-labelled sample, with NLI measuring consistency with retrieved incident types rather than direct visual support. Evaluation is on recorded video; no real-time guarantee is claimed.\n")
A("## 8. Conclusion\n")
A("We built and evaluated a weakly-supervised CNN–Transformer anomaly detector with temporal localisation, calibrated confidence with human-review routing, statistically tested ablations and an evidence-groundedness check for explanations. " + ("" if ev else NA) + "\n")
A("## References\n")
refs = ["A. Vaswani et al., “Attention Is All You Need,” NeurIPS, 2017.",
 "W. Sultani, C. Chen, M. Shah, “Real-world Anomaly Detection in Surveillance Videos,” CVPR, 2018.",
 "Y. Tian et al., “Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning,” ICCV, 2021.",
 "A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML, 2021.",
 "J. Li, D. Li, S. Savarese, S. Hoi, “BLIP-2,” ICML, 2023.",
 "L. Zanella et al., “Harnessing Large Language Models for Training-free Video Anomaly Detection,” CVPR, 2024.",
 "Z. Ding et al., “SlowFastVAD,” arXiv:2504.10320, 2025.",
 "P. Wu et al., “VadCLIP,” AAAI, vol. 38, no. 6, 2024.",
 "H. Lv, Q. Sun, “Video Anomaly Detection and Explanation via Large Language Models,” arXiv:2401.05702, 2024.",
 "A. Acsintoae et al., “UBnormal,” CVPR, 2022.",
 "J. Hessel et al., “CLIPScore,” EMNLP, 2021.",
 "P. He, J. Gao, W. Chen, “DeBERTaV3,” arXiv:2111.09543, 2021; A. Williams et al., “MultiNLI,” NAACL-HLT, 2018.",
 "C. Guo et al., “On Calibration of Modern Neural Networks,” ICML, 2017.",
 "A. Gretton et al., “A Kernel Two-Sample Test,” JMLR, vol. 13, 2012.",
 "J. Johnson, M. Douze, H. Jégou, “Billion-scale Similarity Search with GPUs,” IEEE Trans. Big Data, 2021."]
for i, r in enumerate(refs, 1): A(f"[{i}] {r}")
out = REPO / "report"; out.mkdir(exist_ok=True)
md = out / "report.md"; md.write_text("\n".join(L), encoding="utf-8")
print("wrote", md)
if shutil.which("pandoc"):
    r = subprocess.run(["pandoc", str(md), "-o", str(out / "report.docx"), "--resource-path", str(P["artifacts"])], capture_output=True, text=True)
    print("wrote report.docx" if r.returncode == 0 else f"pandoc failed: {r.stderr[:200]}")
else:
    print("pandoc not found: open report.md in Word/Typora or install pandoc for a .docx")
