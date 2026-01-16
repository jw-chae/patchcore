# PatchCore From Scratch (Debug Log + Final Recipe)

This document records the full debug process and the final, working settings
that match the official PatchCore (patchcore-inspection) results.

All paths and commands below are from this machine.

---

## Final target (official-aligned)

Goal: match patchcore-inspection (Amazon official) as closely as possible.

Final config:
- `configs/benchmark/wrn50_patchcore_official_1p.yaml`

Final mean metrics (15 categories, MVTec):
- PatchCoreX: image AUROC 0.989775, pixel AUROC 0.978916
- patchcore-inspection: image AUROC 0.989456, pixel AUROC 0.979765
- Delta (PCX - inspection): image +0.000319, pixel -0.000849

Result files:
- `runs/mvtec/<category>/patchcore_wrn50_official_1p/seed0/metrics.json`

---

## Final pipeline (official-aligned)

- Backbone: WideResNet-50-2 (ImageNet pretrained)
- Layers: `layer2 + layer3`
- Patchify: patchsize=3, stride=1 (PatchCore patch maker)
- Embedding mapping:
  - pretrain_embed_dimension=1024 (MeanMapper)
  - target_embed_dimension=1024 (Aggregator)
- Coreset: approx greedy coreset, 1%
  - starting points seeded for reproducibility
- Scoring: kNN, k=1, image score = patchcore max reduction
- Preprocess: Resize 256 -> CenterCrop 224 -> Normalize (ImageNet)
- Postprocess: Gaussian blur sigma=4 (scipy.ndimage)
- Metrics: sklearn-based AUROC/AP (same as patchcore-inspection)

---

## Final config (official-aligned)

File: `configs/benchmark/wrn50_patchcore_official_1p.yaml`

Key fields:
- `dataset.resize: 256`
- `dataset.img_size: 224`
- `features.patchsize: 3`
- `features.pretrain_embed_dimension: 1024`
- `features.target_embed_dimension: 1024`
- `memory.type: approx_greedy_coreset`
- `memory.percentage: 0.01`
- `scoring.type: knn_l2`
- `scoring.num_neighbors: 1`
- `scoring.image_score: patchcore`
- `postprocess.blur_sigma: 4`
- `inference.type: faiss_gpu`

---

## Repro safety checks (must be fixed to avoid drift)

1) Resize / interpolation / antialias
   - Resize uses PIL backend via torchvision transforms.
   - Interpolation: BILINEAR.
   - Antialias: keep torchvision default for your version; if changing torchvision, pin the value explicitly.
   - File: `patchcorex/data/transforms.py`

2) Patchify (PatchMaker) details
   - Patch size 3, stride 1, padding = 1 (same as patchcore-inspection).
   - Unfold order matches `PatchMaker` in patchcore-inspection.
   - Align is done before patchify for base features, then patchify per-layer, then align to ref patch grid.
   - File: `patchcorex/models/feature_extractors/wrn_multilayer.py`

3) MeanMapper / Aggregator exact ops
   - MeanMapper: reshape to [N, 1, D] then AdaptiveAvgPool1d to 1024 (no linear/conv, no BN, no activation).
   - Aggregator: reshape to [N, 1, *] then AdaptiveAvgPool1d to 1024.
   - File: `patchcorex/models/feature_extractors/wrn_multilayer.py`

4) Distance metric / normalize location
   - No L2 normalization applied to features for FAISS.
   - FAISS backend uses IndexFlatL2; distances are squared L2 (FAISS default).
   - File: `patchcorex/inference/faiss_gpu.py`

5) Coreset RNG / dataloader order
   - Approx greedy coreset starting points use seeded NumPy RNG.
   - Train dataloader uses shuffle=False; for strict determinism consider num_workers=0.
   - File: `patchcorex/memory/builders/approx_greedy_coreset.py`

6) Gaussian blur definition
   - Use `scipy.ndimage.gaussian_filter` with default mode unless pinned.
   - For absolute reproducibility, fix mode and truncate explicitly if you diverge:
     `gaussian_filter(x, sigma=4, mode="nearest", truncate=4.0)`
   - File: `patchcorex/postprocess/maps.py`

7) Image score definition (no reweight)
   - Patch score: `s_patch[i] = min_j ||q_i - bank_j||_2`
   - Image score: `s_img = max_i s_patch[i]`
   - Implemented via `image_score: patchcore` in config.

8) Results comparison protocol
   - Compare unweighted mean over the 15 categories.
   - Use the same `results.csv` from patchcore-inspection:
     `paper_codes/patchcore-inspection/results/project/group/results.csv`
   - Validate per-category AUROC before averaging.

---

## Code locations (critical)

- Preprocess / resize-crop: `patchcorex/data/transforms.py`
- Dataset resize plumbing: `patchcorex/data/datasets/mvtec.py`, `patchcorex/data/loaders.py`
- Feature extractor + patchify + embed mapping: `patchcorex/models/feature_extractors/wrn_multilayer.py`
- Coreset (approx greedy): `patchcorex/memory/builders/approx_greedy_coreset.py`
- Scoring (kNN): `patchcorex/scoring/knn.py`
- KNN backend: `patchcorex/inference/faiss_gpu.py`
- Image score override: `patchcorex/cli/eval.py`
- Postprocess blur: `patchcorex/postprocess/maps.py`
- Metrics (sklearn): `patchcorex/eval/metrics.py`

---

## Full debug timeline (what was wrong and what fixed it)

1) Initial mismatch (reweight/avgpool path)
   - Config: reweight, avgpool ON, no patchify.
   - Mean image AUROC: 0.981644
   - Mean pixel AUROC: 0.978704
   - Observation: close but not official; strong per-class gaps (capsule, screw, toothbrush).

2) Official-aligned config created (kNN only)
   - New config: `wrn50_patchcore_official_1p.yaml`.
   - Switched to kNN (k=1), no reweighting.
   - Mean image AUROC: 0.977216
   - Mean pixel AUROC: 0.967225
   - Observation: worse; missing PatchCore patchify + embed mapping.

3) PatchCore patchify + embed mapping added
   - Added patchify (patchsize=3) and embed mapping:
     pretrain_embed_dimension=1024 -> target_embed_dimension=1024.
   - Implemented in `wrn_multilayer.py`.
   - Mean image AUROC: 0.980863
   - Mean pixel AUROC: 0.979905
   - Observation: big recovery, but toothbrush/screw still low.

4) Coreset algorithm aligned
   - Switched memory builder to approx greedy coreset.
   - Added seed control for starting points.

5) Preprocess fixed (critical)
   - patchcore-inspection uses Resize 256 -> CenterCrop 224.
   - Added `dataset.resize: 256` and loader support.
   - This fixed the large AUROC gap for toothbrush/screw.
   - Mean image AUROC: 0.988512
   - Mean pixel AUROC: 0.978501

6) Metrics aligned
   - Switched to sklearn-based AUROC/AP (patchcore-inspection style).

7) Image score aligned
   - Forced image score to patchcore max-reduction (patch scores -> image score).

8) Blur aligned
   - Switched Gaussian blur to `scipy.ndimage.gaussian_filter`.
   - Mean image AUROC: 0.989775
   - Mean pixel AUROC: 0.978916

9) Final re-run
   - Mean differences are within ~0.001 for both image and pixel AUROC.

---

## Repro commands

Run a single category:

```bash
PYTHONPATH=/media/jjack/Extreme SSD/paper_codes/patchcore \
conda run -n ad_env python3 -m patchcorex.cli.train \
  --config /media/jjack/Extreme SSD/paper_codes/patchcore/configs/benchmark/wrn50_patchcore_official_1p.yaml

PYTHONPATH=/media/jjack/Extreme SSD/paper_codes/patchcore \
conda run -n ad_env python3 -m patchcorex.cli.eval \
  --config /media/jjack/Extreme SSD/paper_codes/patchcore/configs/benchmark/wrn50_patchcore_official_1p.yaml
```

Run all 15 categories:

```bash
CFG="/media/jjack/Extreme SSD/paper_codes/patchcore/configs/benchmark/wrn50_patchcore_official_1p.yaml"
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
for c in $CATS; do
  tmp=$(mktemp)
  cp "$CFG" "$tmp"
  sed -i "s/^  category: .*/  category: $c/" "$tmp"
  PYTHONPATH="/media/jjack/Extreme SSD/paper_codes/patchcore" conda run -n ad_env python3 -m patchcorex.cli.train --config "$tmp"
  PYTHONPATH="/media/jjack/Extreme SSD/paper_codes/patchcore" conda run -n ad_env python3 -m patchcorex.cli.eval --config "$tmp"
  rm -f "$tmp"
done
```

---

## Verification against patchcore-inspection

Official results file:
- `paper_codes/patchcore-inspection/results/project/group/results.csv`

Compare means:
- PatchCoreX mean (from `metrics.json` files) should be within ~0.001.

---

## Common mistakes (do not repeat)

- Using reweighting or avgpool when matching official PatchCore.
- Missing Resize 256 -> CenterCrop 224.
- Missing patchify + embed mapping (1024/1024).
- Using torch gaussian blur instead of scipy version.
- Unseeded approx greedy coreset starting points.
