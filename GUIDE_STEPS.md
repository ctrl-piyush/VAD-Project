# GUIDE: Train + Test kaise karna hai (step by step)

Poora pipeline **ek baar** chalana hai. Sabse lamba kaam **feature extraction** hai (videos ek baar padhe jaate hain, features
disk par cache ho jaate hain). Uske baad training/tuning/evaluation cached features par hoti hai, isliye dobara train karna sasta hai.
Har stage resumable hai: beech mein band ho jaye to **wahi command dobara chalao**, jo ho chuka hai woh skip ho jayega.

> Agar tum Kaggle ka frames wala dataset (odins0n) use kar rahe ho, to **KAGGLE_GUIDE.md** padho; Step 1-3 wahan alag hain.

## Step 0: Setup (ek baar)
```
python -m venv .venv ; source .venv/bin/activate          # Windows: .venv\Scripts\activate
# pytorch.org se apne CUDA wala torch + torchvision install karo, phir:
pip install -r requirements.txt
pip install transformers pillow                           # CLIP ablation + explanations ke liye (zaroori hai poore synopsis ke liye)
pytest -q                                                 # sab green hona chahiye (2 API/app tests skip honge, ok)
python scripts/check_hardware.py                          # GPU/VRAM/RAM/disk dikhata hai
```
GPU nahi hai to Kaggle/Colab (free GPU) par sirf Step 3 (extraction) chalao, phir `artifacts/dataset/` aur
`artifacts/features/` folders apne laptop par copy karo. Training/evaluation ke liye videos ki zaroorat nahi, sirf manifest + features +
annotation txt files chahiye.

## Step 1: Dataset (manual)
1. UCF-Crime official source se download karo (Sultani et al., CRCV UCF) + 3 annotation txt files.
2. Rakho: `data/ucf_crime/videos/...mp4` aur `data/ucf_crime/annotations/{Anomaly_Train.txt, Anomaly_Test.txt, Temporal_Anomaly_Annotation_for_Testing_Videos.txt}`
   (ya `export VAD_DATA_ROOT=/path` / `VAD_ARTIFACTS=/path`).
3. `Temporal_...txt` kholkar dekho: format `name class s1 e1 s2 e2` hona chahiye. Alag ho to `parse_temporal()` (vad/data/validate.py) badlo.

## Step 2: Chhota test (10 minute, zaroor karo)
```
python -m vad.data.validate --limit 20          # partial files likhta hai, asli manifest ko nahi chhuta
python -m vad.data.validate                     # FULL. exit code 0 aur validation_report.json me problems khali hone chahiye
python -m vad.data.make_val_split
python -m vad.data.features --limit 10          # extraction_report.json me seconds_per_new_video dekho
```
**Time andaza:** `seconds_per_new_video x ~1900 videos` = poore extraction ka time. Agar bahut zyada hai: GPU/Colab use karo, ya
`--set preprocess.batch_clips=16` (GPU par). Akhri option (sir ko batana padega): `--set preprocess.clip_stride=32` (har doosra clip) time aadha kar deta hai.

## Step 3: Poora pipeline ek command me
```
python scripts/run_all.py --seeds 1 2 3 --with-explanations
```
Ye in stages ko order me chalata hai: validate, split, features (3D-CNN), clipfeatures (ablation), analysis, tune, final training
(baseline + CLIP-Transformer + hybrid, har ek 3 seeds), operating point, evaluate (test par EK baar), explanation store + CSV, report.
Time bachana ho: `--no-tune` (default hyper-parameters). Fail ho to error padho, theek karo, **same command dobara**.

## Step 4: Manual kaam jo code nahi kar sakta
1. `artifacts/explanations/explanations_to_label.csv` kholo. Do log (tum aur ek teammate) alag-alag `annotator1`, `annotator2` columns me
   `1` (frame se supported) ya `0` (nahi) likho. Frame image `artifacts/explanations/frames/<video_id>.jpg`.
2. `python -m vad.explain.evaluate_checker --csv artifacts/explanations/explanations_to_label.csv`
3. `python scripts/make_report.py`   -> `report/report.md` aur `report/report.docx` (asli numbers ke saath)
4. `streamlit run app/streamlit_app.py` (optional backend: `uvicorn api.main:app --port 8000`, phir sidebar me URL daalo)

## Step 5: Results ko kaise padhna (important)
- `artifacts/reports/eval_tf.json`: frame-level AUC (paper se comparable) aur video-level AUC **alag** hain.
- Agar hybrid, baseline se behtar **nahi** nikla, to wahi report karo. Ablation ka matlab hi sach dikhana hai. Report ka Discussion section isi ke liye hai.
- Test sirf final evaluate me use hota hai. Test dekhkar hyper-parameter mat badlo (data leakage / galat science).
- Tuning me 0.01 AUC ka fark noise hai (1 seed). Final faisla 3-seed mean ± SD aur bootstrap CI se.

## Common errors
| Error | Fix |
|---|---|
| `CUDA out of memory` | extraction khud batch aadha karta hai; phir bhi ho to `--set preprocess.batch_clips=2` |
| `X videos have no cached features` | extraction poora nahi hua: `python -m vad.data.features` dobara |
| `ambiguous`/`missing` in validation | video ka naam/jagah check karo; same filename 2 folders me nahi hona chahiye |
| frame AUC = nan | temporal annotation file nahi mili ya format alag hai |
| CLIP stage fail | `pip install transformers pillow`, internet chahiye (model download); ablation ke bina bhi baaki chalega |
