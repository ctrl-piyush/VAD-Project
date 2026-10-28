# Kaggle dataset (odins0n/ucf-crime-dataset) ke saath kaise chalana hai

## 1. Pehle samjho ye dataset kya hai (important)
Ye **videos nahi hain**. Official UCF-Crime videos se **har 10th frame, 64×64 PNG** nikaal kar rakhe gaye hain
(14 folders = 13 anomaly + Normal; train ~1.27 million images, test ~111 hazaar images, dataset page ke description ke mutabiq).

Synopsis se jo farak aayega (sir ko batana zaroori hai):
| Synopsis | Is dataset ke saath |
|---|---|
| 16 **consecutive** frames ka snippet (~0.5 s) | 16 **sampled** frames = 160 original frames (~5 s). Motion information kaafi kam, 64×64 ko 112×112 tak upscale |
| Localisation ~0.5 s resolution | ~5 s resolution (stride 16) ya ~2.7 s (stride 8, `--set preprocess.clip_stride=8`) |
| Literature ke AUC se comparable | **Nahi**. VadCLIP jaise 88% numbers se compare mat karo |
| BLIP-2 explanations | 64×64 frames par quality kam hogi |

Jo cheez isme **nahi** hai: **temporal annotation file** (`Temporal_Anomaly_Annotation_for_Testing_Videos.txt`). Ye official UCF-Crime
resources (ya GitHub repos jo isse use karte hain) se alag lena padega, aur `data/ucf_crime/annotations/` me rakhna hai. Iske bina sirf
**video-level AUC** milega; frame-level AUC aur temporal IoU skip ho jayenge (report me aisa likha aayega).

## 2. Kahan chalao
**Best: Kaggle notebook** (free GPU, dataset download nahi karna): Add Data → ye dataset, annotation file ko apne private Kaggle dataset
me upload karke add karo. Dataset `/kaggle/input/...` me milega; folder ka asli naam `ls /kaggle/input` se dekho.
Ya laptop par: `pip install kaggle`, Kaggle API token, `kaggle datasets download -d odins0n/ucf-crime-dataset` (size check karo, 14 lakh PNG hain).

## 3. Commands
```
export VAD_CONFIG=configs/kaggle_frames.yaml            # Windows: set VAD_CONFIG=configs\kaggle_frames.yaml
export VAD_FRAMES_ROOT=/path/jahan_Train_aur_Test_folder_hain
ls $VAD_FRAMES_ROOT                                       # Train/ aur Test/ dikhne chahiye
python -m vad.data.index_frames --root $VAD_FRAMES_ROOT --limit 5     # inspection: filename_examples dekho
python -m vad.data.index_frames --root $VAD_FRAMES_ROOT               # poora; exit code 0 chahiye
python -m vad.data.make_val_split
python -m vad.data.features --limit 10                    # seconds_per_new_video dekho
python scripts/run_all.py --seeds 1 2 3 --frames-root $VAD_FRAMES_ROOT [--no-tune]
python scripts/make_report.py
```
**Gate (rukna padega agar):**
- `filename_examples` me naam `<VideoName>_<number>.png` jaisa nahi hai, ya `unparsable_filename` aaya, to `ls Train/Abuse | head` ka output bhejo, main parser theek kar dunga.
- `warnings` me "official split is 1610/290" aaye: is copy ka split official nahi ho sakta. Train mat karo, pehle mujhe batao (data leakage / galat protocol ka khatra).
- Train/Test overlap (`TRAIN_TEST_OVERLAP`) aaye to bhi ruko.

## 4. Kya automatic hai
- Frames ko video-wise group karke sorted order me 16-16 ke snippets (`vad/data/index_frames.py`).
- Annotation ke original frame numbers ko 10 se divide karke extracted-frame index banana (`data.gt_divisor: 10`); synthetic test me verify hua.
- Time/seconds correct aate hain (effective fps = 30/10 = 3). Maine 30 fps maan liya (UCF-Crime ke videos ~30 fps); alag ho to `--orig-fps`.
- Extraction ~86 hazaar snippets (raw video ke ~8.6 lakh se 10 guna kam), isliye video-route se kaafi tez hona chahiye. Asli speed `--limit 10` se napo.
- Dashboard me video upload karoge to wo bhi har 10th frame + 64×64 resize karke wahi pipeline se jayega (approximate domain match).

## 5. Verify kya hua, kya nahi
Synthetic Kaggle-jaise fixture par poora flow test hua (index → split → features (resume) → train → evaluate, GT mapping, video-vs-frame source equality).
**Asli dataset par nahi chala:** folder layout / filename pattern meri assumption hai (upar gate dekho), 64×64 PNG par asli accuracy/speed unknown.
