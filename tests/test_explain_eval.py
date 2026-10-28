import numpy as np, pandas as pd
from vad.explain.evaluate_checker import evaluate


def test_checker_eval_on_synthetic_labels():
    rng = np.random.RandomState(0); rows = []
    for v in range(40):
        for var in ["rag", "vlm", "rag_vlm"]:
            good = rng.rand() < 0.5
            g = float(np.clip((0.75 if good else 0.25) + rng.randn() * 0.1, 0, 1))
            lab = int(good); flip = rng.rand() < 0.1
            rows.append(dict(video_id=f"v{v}", variant=var, clip_percentile=g, nli=g, groundedness=g,
                             annotator1=lab, annotator2=lab if not flip else 1 - lab))
    rep = evaluate(pd.DataFrame(rows))
    assert rep["cohen_kappa"] > 0.6 and rep["n_disagreements_dropped"] > 0
    c = rep["checker"]["min(clip,nli) [proposed]"]
    assert c["roc_auc"] > 0.9 and c["precision_ungrounded"] > 0.8 and c["recall_ungrounded"] > 0.8
    assert rep["with_evidence_grounding"]["hallucination_rate_after"] < rep["with_evidence_grounding"]["hallucination_rate_before"]
