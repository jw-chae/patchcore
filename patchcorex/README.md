# PatchCoreX README

이 폴더는 PatchCore 기반 이상탐지 코드의 핵심 구현을 담고 있습니다. 학습(메모리뱅크 생성), 평가, 프로파일링, 스윕 설정 생성까지 최소 파이프라인이 포함되어 있습니다.

## 전체 구조와 역할

### 진입점 (CLI)
- `patchcorex/cli/train.py`: 학습용 특징 추출 후 메모리 뱅크 저장.
- `patchcorex/cli/eval.py`: 저장된 메모리 뱅크로 평가, 지표/시각화 생성.
- `patchcorex/cli/profile.py`: 단일 배치 기준 추론 시간/VRAM 측정.
- `patchcorex/cli/sweep.py`: 하이퍼파라미터 조합별 config 생성.
- `patchcorex/cli/options.py`: 레지스트리에 등록된 옵션/스키마 출력.

### 핵심 엔진/레지스트리
- `patchcorex/engine.py`: feature extractor / memory builder / scorer / inference backend 생성.
- `patchcorex/utils/registry.py`: 동적 레지스트리(플러그인 방식 확장 지점).

### 데이터/전처리
- `patchcorex/data/datasets/mvtec.py`: MVTec AD 데이터셋 로더.
- `patchcorex/data/loaders.py`: DataLoader 빌드.
- `patchcorex/data/transforms.py`: 이미지/마스크 변환.

### 특징 추출/모델
- `patchcorex/models/feature_extractors/*`: 패치/스크린 특징 추출기.
- `patchcorex/models/backbones/*`: backbone 모듈 (OpenCLIP, DINOv2/v3, ConvNeXt, Swin 등).

### 메모리 뱅크
- `patchcorex/memory/bank.py`: 단일 메모리 뱅크.
- `patchcorex/memory/dual_bank.py`: dual 모드용 메모리 뱅크.
- `patchcorex/memory/builders/*`: 샘플링/축소 전략 (full, kcenter, random, reservoir, rsw_e, rrsw_e).

### 스코어링/추론/후처리
- `patchcorex/inference/*`: KNN/FAISS 백엔드.
- `patchcorex/scoring/*`: kNN, Mahalanobis, position-aware, manifold 등.
- `patchcorex/postprocess/*`: patch -> pixel map 변환, blur, top-k pooling, threshold.
- `patchcorex/viz/*`: overlay/heatmap, CAM 시각화.

## 동작 모드

### 1) PatchCore 기본 모드 (Single Bank)
- 특징 추출기가 단일 패치 특징만 반환.
- `memory_bank.pt` 저장/로드.
- `inference`, `scoring`, `memory` 설정은 단일 구조.

### 2) Dual Bank 모드
- 특징 추출기가 `{ "seg": ..., "scr": ... }`를 반환.
- `dual_memory_bank.pt` 저장/로드.
- `memory.seg`, `memory.scr` 각각 정의.
- `inference.seg`, `inference.scr` / `scoring.seg`, `scoring.scr` 각각 정의.
- `scoring.use_seg_image`를 켜면 이미지 스코어는 seg 결과로 사용.

`eval.py`/`profile.py`는 `dual_memory_bank.pt`의 존재 여부로 모드를 자동 판별합니다.

## 실행 흐름 (기본)

1) 학습 (메모리 뱅크 생성)
```bash
python -m patchcorex.cli.train --config configs/your_config.yaml
```
출력:
- `runs/<dataset>/<category>/<exp>/seed<seed>/memory_bank.pt` 또는 `dual_memory_bank.pt`
- `bank_stats.json`
- `config.yaml` (복사본)

2) 평가
```bash
python -m patchcorex.cli.eval --config configs/your_config.yaml
```
출력:
- `metrics.json`
- 옵션에 따라 `qualitative/` 시각화 이미지

3) 프로파일링
```bash
python -m patchcorex.cli.profile --config configs/your_config.yaml
```
출력:
- `profile.json` (timings, VRAM)

4) 스윕용 config 생성
```bash
python -m patchcorex.cli.sweep --config configs/your_sweep.yaml --out-dir runs/sweeps
```
출력:
- `runs/sweeps/config_0000.yaml` 등
- `runs/sweeps/index.csv`

5) 옵션/스키마 확인
```bash
python -m patchcorex.cli.options --format json
```

## 제공되는 configs

이 레포에는 실험에 사용된 config 파일들이 포함되어 있습니다.
- 경로: `configs/`
- 예시 실행:
```bash
python -m patchcorex.cli.train --config configs/example.yaml
python -m patchcorex.cli.eval --config configs/example.yaml
```

## 공식 PatchCore 대비 체크리스트

아래 항목들은 공식 PatchCore(논문/원 저자 코드) 기준으로 성능/속도에 영향을 줄 수 있는 차이점입니다.
- 전처리에서 `GaussianBlur`를 쓰지 않음 (공식은 Resize → CenterCrop).
- KNN은 FAISS 기반 사용을 전제로 함 (PyTorch `cdist`는 느리고 수치 차이 가능).
- Memory bank는 FP32 기준이 기본 (FP16은 속도/메모리 절약 대신 정밀도 손실 가능).

## 설정 파일 구조 (YAML)

### 공통 키
```yaml
experiment:
  name: exp_name
  seed: 0

dataset:
  name: mvtec
  root: /path/to/mvtec
  category: bottle
  img_size: 224
  batch_size: 8
  num_workers: 2

features:
  type: vit_patches
  # type별 옵션은 options 출력 참고

backbone:  # 필요할 때만
  type: openclip_vit
  model_name: ViT-L-14
  pretrained: auto

memory:
  type: kcenter
  K: 10000
  seed: 0
  max_samples: 200000
  dtype: fp16

inference:
  type: faiss_gpu

scoring:
  type: knn_l2
  k: 1
  image_agg: max

postprocess:
  blur_sigma: 0
  topk_pool:
    kernel: 3
    topk: 3
  threshold:
    type: fpr_at_tpr
    tpr: 0.99

eval:
  pixel_fpr_limit: 0.3

viz:
  enabled: true
  topk: 8
  alpha: 0.5

cam:
  enabled: false
  topk: 8
  alpha: 0.5
```

### Dual 모드 예시
```yaml
features:
  type: vit_dual
  seg_normalize: l2
  scr_normalize: l2
  scr_source: cls

memory:
  seg:
    type: kcenter
    K: 10000
    seed: 0
    max_samples: 200000
    dtype: fp16
  scr:
    type: full
    dtype: fp16

inference:
  seg: { type: faiss_gpu }
  scr: { type: faiss_gpu }

scoring:
  seg: { type: knn_l2, k: 1, image_agg: max }
  scr: { type: knn_l2, k: 1, image_agg: max }
  use_seg_image: false
```

## 옵션/구현 팁
- `position_aware` 스코어는 `positions`가 필요하므로 패치 위치가 계산되는 extractor를 사용하세요.
- `rsw_e`, `rrsw_e` 메모리 빌더는 per-image 입력을 요구합니다.
- FAISS 백엔드를 사용하려면 FAISS가 설치되어 있어야 합니다.
- `options` 출력에 스키마 예시가 포함되어 있어 config 작성에 유용합니다.

## 출력 디렉터리 규칙
- 학습/평가/프로파일 공통으로 `runs/<dataset>/<category>/<exp>/seed<seed>/`에 저장됩니다.
- `--run-dir`나 `--bank-dir`로 경로를 바꿀 수 있습니다.
