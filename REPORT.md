# Assignment 2 Report

- **Name**: Paul Khairallah
- **Student ID**: 18966
- **Email**: pkhairallah.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

**The decision.** A Dutch packhouse washes table potatoes and packs them into
3 kg retail bags. A camera sits on one ~6 t/h lane after upstream destoning and
washing, just before final packing. For every object on the belt the model
decides: **pass**, **eject** (air jet into the reject stream), or, if a review
band is used, **divert** to a buffered side station where one person checks it.
The line supervisor owns the reject setting; quality staff own release
criteria and the investigation of any foreign material that escapes. The unit
of decision is one object on the belt, not one image.

This is an assumed business setting, not an identified customer site. Upstream
destoning is standard on such lines (Tong's destoners remove stones before
washing, at up to 100 t/h,
[source](https://tongengineering.com/us/product/vegetable-destoners/)), which is
why stones are assumed rare at this camera.

**What each mistake costs** (EUR, per object):

| Outcome | Cost | Construction | Evidence status |
|---|---|---|---|
| Missed defect (FN) | **€0.77** | 20% chance that a bag with a rotten/blemished potato triggers a €3.85 credit | Bag price anchored; claim probability and severity mix assumed |
| Missed stone (FN) | **€19.89** | 50% chance of a routine claim × (€3.85 credit + 0.75 h × €47.90/h handling) | Handling time and claim probability assumed; not a measured incident cost |
| Good potato rejected (FP) | **€0.060** | 0.200 kg × (€0.34/kg commodity value − €0.04/kg salvage) | Commodity anchor checked; mass and salvage assumed |
| One human review | **€0.0499** | €47.90/h × 3 s ÷ (3,600 × 0.80 utilisation) | Hourly anchor checked; time and utilisation assumed |
| Correct reject / correct pass | €0 | A rejectable item has no saleable value; costs common to every object cancel | Simplification |

Anchors, checked on 28 October 2026 (the LNCN quote page is dynamic, so the week used is stated):

- **Labour:** Eurostat puts 2025 hourly labour cost in the Netherlands at
  €47.90 (whole economy, employers with 10+ staff, wage and non-wage costs).
  [Eurostat](https://ec.europa.eu/eurostat/en/web/products-eurostat-news/w/ddn-20260331-2)
- **Potato value:** LNCN Middenmeer firm-cooking 40/65 quotation, 2026 week 37,
  €33–35 per 100 kg; €0.34/kg is the midpoint. A field-crop quote, not packed
  revenue. [LNCN](https://www.landbouwbeurs.com/beursnoteringen/tafelaardappelen),
  [NAO](https://nao.nl/marktinfo/beursprijzen-middenmeer/)
- **Bag credit:** Albert Heijn's regular price for a 3 kg bag of firm-cooking
  potatoes is €3.85 (€1.28/kg); AH's quality guarantee makes refund a real
  mechanism but says nothing about supplier chargebacks.
  [AH product](https://www.ah.nl/producten/product/wi96050/vastkokende-aardappelen),
  [AH guarantee](https://over.ah.nl/overzicht/kwaliteit-en-productveiligheid)
- **Stones as foreign material:** NVWA requires risk assessment and measures
  for foreign material but gives no euro value per stone.
  [NVWA](https://www.nvwa.nl/onderwerpen/voedselveiligheid/productvreemde-materialen-in-levensmiddelen)

**The asymmetry.** A missed defect costs about **12.8×** a false reject
(€0.77 / €0.06) and a missed stone about **331×**. Recall on `bad` and `stone`
therefore matters far more than precision, and the threshold should sit low.
These figures exclude injury, recall and lost contracts, so the stone cost is a
floor, not a full foreign-body risk price.

**Base rates and line.** 1.5% of incoming objects are defective (above the test
belt's ~1.0%, to represent a somewhat poorer lot) and 0.05% are stones (one per
2,000 after destoning). Both are planning assumptions, not measured production
rates. The lane runs 30,000 objects/h (~6 t/h at 200 g) for one 8-hour shift,
with at most one reviewer at 3 s per check and 80% utilisation. A reviewer is
assumed to miss 15% of real defects/stones and to bin 3% of good potatoes.

## Dataset

- **Roboflow Universe:** workspace `ms-workspace-m1gci`, project
  `potato-sorting-on-conveyor`, **version 9**
  (https://universe.roboflow.com/ms-workspace-m1gci/potato-sorting-on-conveyor/dataset/9),
  CC BY 4.0. Downloaded as is in YOLOv11 format; not forked.
- **What one image is:** a 512×512 top-down frame of potatoes on a roller
  conveyor, ~30 objects per frame. Two conveyor cameras (file prefixes
  `kartoffel_auto` and `BELANA_auto`) plus ~50 web/phone photos in train.
- **Classes:** `potato` (every potato), `bad` (a box around a rot/blemish spot
  **on** a potato; the potato itself is also boxed as `potato`), `stone`, and
  `cut` (a cut-open potato; almost all from web photos).
- **Why it serves the problem:** it is the actual sorting scene, the actual
  camera geometry, and the two things a sorter removes (defects and stones),
  rather than studio photos of single potatoes. Because `bad` marks the spot, not
  the potato, I turn detections into one decision per object: a potato is
  "defective" if a `bad` box sits inside it, and its reject score is the
  confidence of the most confident predicted `bad`/`cut` box inside it (or of an
  overlapping predicted `stone` box).

## What the dataset's composition implies

| Split | Images | potato | bad (images with) | stone (images with) | cut |
|---|---|---|---|---|---|
| train | 3,012 (≈1,004 originals × 3 augmented copies) | 84,514 | 1,160 (772) | 2,075 (279) | 176 |
| valid | 121 | 3,763 | 58 (30) | 43 (9) | 10 |
| test | 99 | 2,953 | 29 (21) | 93 (8) | 3 |

Object size (side of the box ÷ side of the image, median): potato 0.10–0.11
(~55 px), stone ~0.09, but **bad only 0.026–0.033 (~13 px)**, and the smallest
10% of blemishes are under 6 px. Version 9 is pre-processed (contrast
stretching, 512×512 stretch) and its training split carries Roboflow's
augmentations (±15% brightness, 90° rotations, ±15° rotation).

What this implies:

1. **The positive classes are the thin ones.** 29 `bad` instances and 3 `cut`
   instances in test. A recall measured on 29 objects has a 95% interval about
   ±17 points wide, so I report Wilson intervals and do not trust any
   difference of a few points. Nothing at all can be concluded about `cut`.
2. **Stones are clustered.** 93 test stones in 8 frames. Stone recall is really
   measured on 8 scenes, not 93 independent events.
3. **Blemishes are tiny.** At ~13 px they sit at the edge of what a YOLO
   detector resolves at 512 px input, which is why I train at 640 (upsampling
   gives the spots more pixels) and why I expect `bad` to be the class that
   drags.
4. **The test split is not an honest stand-in for production.** The conveyor
   frames carry capture timestamps: **94 of the 99 test frames were taken
   within 2 seconds of a training frame** (median gap 0.56 s), on the same two
   days. They show the same potatoes on the same belt from almost the same
   position. Test scores will therefore be optimistic about a new day, a new
   batch or a new camera. The assignment asks me to use the split as published,
   so I keep it, but every number below should be read as "on frames very like
   training", not as a production estimate.

## Finetuning and baselines

**Model and task.** Object detection, `yolo11s` finetuned from the COCO
checkpoint. Detection rather than classification because a frame holds ~30
objects and the sorter has to act on each one, and because the defect label is a
small spot on a potato, which a whole-image classifier could not localise.
`yolo11s` rather than `yolo11n` because the bottleneck is ~13 px blemishes and
the extra capacity is cheap on a GPU.

**Where and how it was trained.** Google Colab, one Tesla T4, via
`colab/train_on_colab.ipynb`, which clones this repo and runs
`python main.py train` unchanged. Settings (all in `TRAIN_CFG` in `train.py`):
60 epochs, imgsz 640, batch 32, optimizer auto (AdamW, lr 0.00125), mosaic
off for the last 10 epochs, extra vertical flips (top-down camera), seed 0,
deterministic. 60 epochs took 47 minutes. The best checkpoint (epoch 48, val
mAP50 0.662 / mAP50-95 0.552) is committed as `weights/potato_yolo11s.pt`. The
evaluation below was re-run on CPU from those committed weights and reproduces
the test numbers Colab printed (mAP50 0.617, mAP50-95 0.501).

**Results on the same test split (99 images).**

| Model | mAP50 | mAP50-95 | Class-agnostic AP50 (finds the objects at all?) | As a sorter at its best threshold, € per 1,000 objects |
|---|---|---|---|---|
| Finetuned yolo11s | **0.617** | **0.501** | 0.995 | **€4.64** (t = 0.03) |
| COCO yolo11s zero-shot | 0.000 | 0.000 | 0.441 | €21.49 (never rejects anything) |
| Predict nothing | 0.000 | 0.000 | 0.000 | €21.49 |

The COCO model scores zero on every class by construction: it has no potato,
stone or blemish class. To give it a fair hearing I also scored it
class-agnostically. It does find 94% of the objects at some confidence, but at
conf 0.25 it calls them `elephant` (11), `hot dog` (7), `knife` (5) and `spoon`
(1). It has no label that means "remove this", so as a sorter it is identical to
doing nothing. The finetuned model beats both baselines by a wide margin, which I
expected; what I did not expect was how badly the blemish class does at box
level (next section) while still being useful for the decision.

## Per-class evaluation and failure modes

Positive class for the decision: **`bad` and `stone`** (things to remove);
`potato` is the negative. `cut` is also a reject reason but has 3 test
instances, so I report it and draw no conclusion from it.

| Class | Test instances | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| potato | 2,953 | 0.989 | 0.998 | 0.995 | 0.971 |
| stone | 93 (8 images) | 0.912 | 0.968 | 0.972 | 0.884 |
| bad | 29 (21 images) | 0.289 | 0.103 | 0.120 | 0.033 |
| cut | 3 | 0.565 | 0.333 | 0.382 | 0.118 |
| **all (mean)** | | 0.689 | 0.601 | **0.617** | **0.501** |

(P and R are Ultralytics' values at the confidence that maximises F1.)

Box-level confusion matrix at conf 0.25 (rows predicted, columns true):

| | true bad | true cut | true potato | true stone | true background |
|---|---|---|---|---|---|
| pred bad | 3 | 0 | 0 | 0 | 6 |
| pred cut | 0 | 1 | 0 | 0 | 0 |
| pred potato | 1 | 1 | 2,947 | 2 | 29 |
| pred stone | 0 | 0 | 0 | 88 | 10 |
| pred background (missed) | 25 | 1 | 6 | 3 | — |

**Who carries the headline number.** The 0.617 mAP50 is an average of potato
(0.995) and stone (0.972) being nearly solved and `bad` (0.120) being close to
useless *as a box detector*: at conf 0.25 it finds 3 of 29 blemishes. Averaging
hides exactly the class the business case depends on.

**Why it still works as a sorter.** The blemish boxes exist, they just come
with low confidence. Scoring each potato by its most confident blemish box, the 31
defective potatoes on the test belt have a median score of 0.048, while 2,251 of
the 2,922 good potatoes score exactly 0 and 90% score below 0.007. The ranking is
good even though the confidences are low (low scores rank well; they are not
calibrated probabilities), which is why the operating threshold
ends up at 0.03 rather than anywhere near Ultralytics' default 0.25. At 0.25 the
sorter would catch only 16% of defects and cost €10.15 per 1,000, more than double the cost at 0.03.

**Where it breaks.**
- **Blemishes.** `bad` boxes are ~13 px and the model is never confident about
  them. At t = 0.03, 6 of 31 defective potatoes still pass. They are not simply
  the smallest spots (median blemish side 15 px for the misses vs 14 px for
  those caught), so size alone does not explain them; 6 cases are too few to
  say what does.
- **Stray reject boxes.** 109 low-confidence blemish/stone boxes land on no
  labelled object. The ejector would still fire, so they count as false
  ejections in the costing. At conf 0.25, 10 stone boxes and 29 potato boxes
  also match no labelled object.
- **Stones** are the strong class: 88 of 93 found, 3 missed, 2 called potato.
  But the 93 come from 8 frames, so this is closer to 8 observations than 93.
- **`cut`** is untestable (3 instances) and comes mostly from web photos that
  look nothing like the belt.

## Cost matrix and operating point

**Method.** Each object on the test belt has one reject score (section above).
Recall and false-reject rates are measured per group on the test belt (31
defective potatoes/cut pieces, 93 stones, 2,922 good potatoes, plus 109 stray
reject boxes) and re-weighted to the production base rates, not the test
split's clustered mix. Stray boxes are extra ejector events: following the
brief they are divided by 2,922 and costed like false rejects. That is a
convention; ejector logs would be needed to tell harmless air pulses from lost
potatoes.

`cost per 1,000 = 1,000 × [π_d (1 − R_d(t)) C_d + π_s (1 − R_s(t)) C_s + π_g F(t) C_fp]`

where F(t) = (good + stray scoring ≥ t) / 2,922. The sweep runs over 104
thresholds from 0.001 to 0.99 using every object's actual score
(`costing.py`, `artifacts/cost_sweep.csv`). I also rebuilt the calculation
independently from the 24-row count table with a standard-library script
(`analysis/recompute.py`); both give the same optimum and the same costs.

The 3,049 labelled objects (2,953 potatoes + 93 stones + 3 cut) become 3,046
decision units because each of the 3 cut pieces is also boxed as a potato and
is counted once, as a cut defect.

![cost against threshold](artifacts/cost_vs_threshold.png)

| Threshold | € / 1,000 | Defect recall | Stone recall | Good potatoes rejected (incl. strays) |
|---|---|---|---|---|
| 0.001 | 15.77 | 100% | 100% | 26.7% |
| 0.01 | 5.93 | 93.5% | 97.8% | 8.4% |
| 0.02 | 4.72 | 87.1% | 96.8% | 4.9% |
| **0.03** | **4.64** | **80.6%** | **96.8%** | **3.5%** |
| 0.05 | 7.62 | 48.4% | 96.8% | 2.3% |
| 0.25 | 10.15 | 16.1% | 96.8% | 0.2% |
| 0.50 | 12.34 | 0% | 92.5% | 0.1% |

**Operating point: reject at a score of 0.03.** Cost **€4.64 per 1,000
objects, €1,113 per 8-hour day**:

| Component | Calculation | € / 1,000 |
|---|---|---|
| Missed defects | 1,000 × 0.015 × 6/31 × €0.77 | 2.235 |
| Missed stones | 1,000 × 0.0005 × 3/93 × €19.8875 | 0.321 |
| False rejects | 1,000 × 0.9845 × 103/2,922 × €0.06 | 2.082 |
| **Total** | | **4.638** |

Per 1,000 incoming objects that projects 2.9 missed defects, 0.016 missed
stones (about **3.9 stones a day** at 240,000 objects) and 34.7 false ejector
events. Defect recall 80.6% has a 95% Wilson interval of 64%–91% (n = 31);
stone recall 96.8% has 91%–99%, but the 93 stones sit in 8 frames, so that
interval is too narrow.

**It is not a large-margin winner.** t = 0.02 costs €4.72 per 1,000, only
€20 a day more. Just above 0.03 the curve is steep: at 0.05 defect recall
falls to 48% and cost rises to €7.62.

**Sensitivity** (one input moved, everything else fixed, full sweep and band
search repeated each time):

| Input (base value) | Range that keeps t = 0.03, no review | Outside that range |
|---|---|---|
| Missed defect C_d (€0.77) | about €0.24–0.86 | above → t = 0.02 (above ~€1.9 → 0.0075); below → 0.12, then 0.21 |
| False reject C_fp (€0.06) | €0.054–0.102 | below → 0.02; above → a review band [0.03, 0.09) appears |
| Defect rate π_d (1.5%) | about 0.46%–1.67% | above → 0.02; below → 0.12, then 0.21 |
| Missed stone C_s (€19.89) | €0–178 | above → 0.0075 |
| Stone rate π_s (0.05%) | 0%–0.445% | above → 0.0075 |
| Review cost C_r (€0.0499) | ≥ €0.0412 | below → band [0.02, 0.03) |
| Reviewer errors h_m, h_f alone | whole 0–100% range | no change |

**The assumption that matters most is the defect side:** the answer moves from
0.03 to 0.02 if defect prevalence rises above about 1.67% or a missed defect
costs more than about €0.86. Both sit close to my chosen values (1.5%, €0.77),
so the first things to measure on a real line are the incoming defect rate and
the actual claim cost per bad bag. At €3.85 per bag the €0.86 limit corresponds
to a claim probability of about 22%; using AH's €2.99 promotional price instead
gives C_d = €0.60, still inside the range. Removing the salvage value
(C_fp = €0.068) keeps t = 0.03 at €4.92 per 1,000.

## What a person still has to check

**Recommendation: nobody, at these costs.** Searching all threshold pairs
(pass below L, review from L to H, reject at H and above), the cheapest policy
is an **empty band, L = H = 0.03**, with or without a staffing cap. Review only
pays if it costs less than about **€0.041 per check**; mine costs €0.0499.

**Why.** The cheapest non-empty band is **[0.02, 0.03)**. On the test belt it
moves 2 defects, 0 stones and 41 good-plus-stray objects into review, so:

`review share = 0.015 × 2/31 + 0.9845 × 41/2,922 = 1.48% of objects`

That is **443 checks/hour, 3,548 a day**, or **0.46 of one person**: feasible
with the one available reviewer (capacity 3,600 × 0.80 / 3 = 960 checks/h,
3.2% of objects). But it costs **€4.77 per 1,000**, more than the model alone:
it saves a few missed defects and false rejects, and pays more in labour.

Break-even, for that band: `C_r* = [0.015 × (2/31) × 0.85 × €0.77 − 0.9845 ×
(41/2,922) × 0.03 × €0.06] / 0.01478 = €0.0412`. Review becomes worthwhile if
labour falls below about **€39.5/h**, a check takes under about **2.5 s**, or
the reviewer is busy more than about **97%** of the time. If review instead
needs one dedicated hire for the shift (€383.20), the best band [0.02, 0.04)
costs €1,341 a day, more than the model alone.

How labour is charged changes the answer: charging only active time
(€47.90 × 3 s / 3,600 = €0.0399 per check) makes [0.02, 0.03) slightly cheaper
(€4.62 per 1,000), saving just €4.44 a day. I use the paid-time convention,
because the reviewer's idle time is paid either way.

**Priced against the alternatives:**

| Policy | € / 1,000 | € / 8-hour day | Review load |
|---|---|---|---|
| Do nothing (also the COCO and predict-nothing baselines) | 21.49 | 5,158 | none |
| A person checks every object | 54.89 | 13,174 | 30,000/h, ~31 people at once: impossible with one |
| **Model alone, t = 0.03** | **4.64** | **1,113** | none |
| Best non-empty band [0.02, 0.03), paid-time labour | 4.77 | 1,144 | 443/h, 0.46 people |
| Same band, active-time labour | 4.62 | 1,109 | 443/h, 0.46 people |
| Band with one dedicated full-shift hire, [0.02, 0.04) | 5.59 | 1,341 | 767/h |

Against doing nothing, the model saves about **€16.86 per 1,000, €4,045 a day**.
That is the most the camera, compute, ejector, maintenance and monitoring could
cost per day before the advantage disappears; it is not net profit.

The review station still has a role as a **fallback**: if the defect rate
climbs or claim costs turn out higher than assumed, the sensitivity table shows
the threshold should drop, and a review band starts to pay once false rejects
cost more than about €0.10. The dashboard lets the supervisor see that live.

## Limitations & next steps

1. **Near-duplicate test split.** 94 of 99 test frames were taken within 2 s of
   a training frame (median 0.56 s), all on the same two days. Test scores are
   optimistic for a new day, lot or camera. Next: collect new capture sessions
   across days, lots, cultivars and lighting; keep each session (and its
   augmented copies) inside one split; choose thresholds on a development set
   and evaluate once on an untouched grouped holdout.
2. **The threshold was tuned on the test counts**, so the test set is now
   decision-tuning evidence, not an untouched final evaluation; and it rests on
   31 defects (each one = 3.2 points of recall). Next: re-run the sweep on a few
   hundred defective potatoes from production before going live.
3. **Ground-truth-assisted scoring.** Object scores attach predicted blemish
   boxes to ground-truth potato boxes. A working line needs detected and
   tracked objects plus verified ejector timing, which can add failures this
   evaluation does not see. Next: measure end-to-end on the line (detection,
   tracking, diversion, ejector logs, final contamination checks).
4. **Blemishes are small and the scores low.** Low confidences rank defects
   well but are not calibrated probabilities. Next: label the whole defective
   potato, or add a second-stage classifier on potato crops; train at higher
   resolution.
5. **Assumed costs and rates.** Claim probabilities, handling time, salvage
   value, defect and stone prevalence, and reviewer accuracy are planning
   assumptions. Next: costed complaint records, a reject-stream recovery quote,
   an object-count prevalence sample, and a timed, blinded review study on the
   hard objects the band would send.
6. **Severe stone outcomes are not priced.** The €19.89 is a routine-claim
   cost; injury, recall or a lost contract would be far larger. Economic
   optimisation cannot set an acceptable stone escape rate: upstream
   foreign-material controls and quality release criteria stay in place.
7. **Queues and bursts.** Average review capacity says nothing about clustered
   arrivals; a review station needs a buffer and a tested overflow/line-stop
   rule rather than a default to pass or eject.
8. **`cut` is not testable** (3 test instances, mostly web photos). Next: drop
   it from the decision or collect cut potatoes on the real belt.

## Generative AI use disclosure

I used Claude throughout this assignment. It helped with writing the code
and formatting of the report.
