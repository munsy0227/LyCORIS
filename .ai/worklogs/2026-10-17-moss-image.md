# MOSS-Image LyCORIS 지원

## 승인된 범위

- LyCORIS 라이브러리의 generic/Kohya 호환 생성 API에서 MOSS 어댑터를 지원한다.
- full-matrix LoKR+DoRA를 우선하고 표준 LoRA도 검증한다.
- `moss`는 DiT 블록의 252개 Linear, `moss-full`은 Transformer 전체 269개
  Linear를 대상으로 한다. 두 범위 모두 제공하기로 사용자가 선택했다.
- sd-scripts 학습기, ComfyUI 로더, 5B 전체 학습과 이미지 품질 평가는 제외한다.
- 구현 후 사용자가 로컬 커밋을 요청했다. 이번 변경만 커밋하고 푸시는 수행하지 않는다.

## 조사 근거

- 모델 경로: `/run/media/munsy0227/MSY EXT4/MOSS-Image-Base`.
- 42개 DiT 블록의 `proj.q/k/v/o/ffn_up_gate/ffn_down`은 252개 Linear다.
- Transformer의 나머지 Linear 17개를 합치면 269개다. 실제 safetensors
  헤더의 모듈 이름과 2차원 weight shape로 확인했다. 가중치 본문은 로드하지 않았다.
- Qwen3.5 텍스트 인코더와 FLUX.2 VAE는 별도 구성 요소다. Transformer의
  `text_embedder`는 projection + text-only block 2개이며 Qwen 자체와 다르다.
- 공식 영문/중문 사용자 가이드는 Qwen 텍스트 인코더 고정을 권장하지만 LoRA
  대상 계층 목록을 명시하지 않는다. 두 프리셋을 공식 LoRA 범위라고 표현하지 않는다.
- 구현 전 구조 모형 확인: generic `full`은 269개를 선택했지만 Kohya `full`은
  MOSS diffusion 모듈을 선택하지 못했다. 새로운 연산 구현보다 명시적인 대상 범위와
  두 API의 일치, 체크포인트/수명주기 검증이 필요한 상황이었다.

## 변경

- `lycoris/config.py`: 앵커를 포함한 원본 경로 정규식으로 두 프리셋을 추가했다.
  `_Block` 등의 흔한 private 클래스명으로 대상을 판정하지 않는다. Norm/Qwen/VAE는
  제외하고 `moss-full`에만 Transformer 내부 text adapter 등 17개 Linear를 추가한다.
- factory는 클래스 전역에 `apply_preset()`을 실행하지 않고 생성자에 프리셋을 전달한다.
  생성자는 기존 클래스 기본값과 프리셋 값을 deepcopy하여 인스턴스에 보관한다.
  클래스 전역을 변경하지 않으므로 성공/실패 모두 후속 생성 호출에 영향을 남기지 않는다.
- 기존 직접 호출용 `apply_preset()` 클래스 메서드는 유지했다. Kohya의 GLoRA 전용
  타깃 조정도 인스턴스 설정에 적용하여 전역 설정을 덮어쓰지 않게 했다.
- generic 체크포인트 복원은 인스턴스 prefix를 사용하며 `preset=...`도 전달할 수 있다.
  custom-prefix 프리셋은 복원 시 다시 전달한다. 기존 safetensors tensor key 형식은 유지했다.
- LoKR/DoRA/LoRA 연산 구현은 변경하지 않았다.
- `docs/usage/moss-image.md`: 로컬 소스/Transformer 로딩, 기본 가중치 고정,
  optimizer 연결, portable/native checkpoint 구분, 복원과 병합 예제를 추가했다.
  `docs/usage/presets.md`에서도 연결한다.
- `test/test_moss.py`: 구조 검사와 로컬 공식 소스 통합 검사를 분리했다.
  로컬 통합 검사는 `LYCORIS_MOSS_MODEL_PATH`로 명시적으로 활성화한다.

## 검증 환경

- `/tmp/lycoris-moss-venv`에 별도 검증 환경을 만들었다. 기존 학습/ComfyUI 환경의
  패키지를 업그레이드하지 않고 기존 PyTorch 런타임을 읽어 사용했다.
- PyTorch 2.14.1+cu130, diffusers 0.41.0, transformers 5.18.0,
  safetensors 0.8.0, pytest 9.1.1.
- NVIDIA GeForce RTX 4070, CPU FP32와 CUDA BF16.
- CPU 스레드 환경 변수는 1로 제한하고 모델/HF 네트워크 접근은 offline으로 제한했다.
- 모델 원본에는 쓰지 않았고 5B/Qwen/VAE 가중치를 로드하지 않았다.

## 통과한 검사

- MOSS 단독: 7 passed, 26 subtests passed.
- 실제 체크포인트 헤더: 정확히 252/269개 대상, 구조 모형에서도 두 API의 실제 대상
  객체와 어댑터 이름이 일치한다. 유사 이름, 중첩된 Qwen/VAE 경로는 제외된다.
- 축소한 실제 공식 MOSS Transformer: dim=64, FFN=128, heads=4, layers=2,
  text_dim=32, freq_dim=32, latent_channels=8. 두 API × 두 프리셋 ×
  LoKR+DoRA/LoRA × CPU FP32/CUDA BF16 = 16개 학습/복원 시나리오.
- 초기 출력 보존, 2회 optimizer step, 유한한 손실/기울기, 출력의 실제 변화,
  기본 파라미터의 불변성과 gradient 부재를 검사했다.
- LoKR unit scale, FP32 magnitude 및 BF16 portable export에서 magnitude FP32 유지를 검사했다.
- 새 축소 모델에서 native state/portable safetensors 복원, multiplier=0/0.5,
  병합 출력과 병합 해제를 검사했다. LoKR undo는 원본과 bitwise 일치한다.
- LoRA는 기존 가산 merge/undo를 사용하므로 undo에도 고정된 수치 기준을 적용했다.
  CPU rtol=1e-5/atol=1e-6, BF16 rtol=0.03/atol=0.03을 사용했으며 오류를
  통과시키기 위해 허용오차를 늘리지 않았다.
- 프리셋 성공/실패 후 클래스 설정 불변, 이후 기존 프리셋 호출, 인스턴스 간
  mutable 설정 분리, custom-prefix 저장/복원을 검사했다.
- 관련 통합 회귀 최종 결과: **190 passed, 3 skipped, 198 deselected,
  180 subtests passed**. 기존 Anima 448개 범위, optimizer, LoKR 체크포인트와
  CUDA, generic wrapper의 관련 사례가 포함된다. 기존 fixture 클래스의
  PytestCollectionWarning 2개가 있으며 실패는 아니다.

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/munsy0227/LyCORIS \
  LYCORIS_MOSS_MODEL_PATH='/run/media/munsy0227/MSY EXT4/MOSS-Image-Base' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/lycoris-moss-venv/bin/python -m pytest \
  test/test_moss.py test/lokr.py test/test_lokr_cuda.py \
  test/test_kohya_optimizer.py test/preset_exclude_name.py test/wrapper.py \
  -k 'moss or lokr or anima or preset or optimizer' -q --tb=short
```

- 변경한 Python 파일 5개의 Black/Ruff 검사, Python AST 검사, 문서 예제 Python
  블록 3개의 구문 검사, `git diff --check`가 통과했다.

## 한계

- 모든 알고리즘의 wrapper 전체 행렬까지 포함한 추가 실행은 55초 실행 제한에
  도달하여 완료 결과가 없다. 위 명령으로 범위에 필요한 회귀를 별도로 완료했으며
  전체 저장소 테스트 통과라고 주장하지 않는다.
- 5B 전체 학습의 VRAM/RAM 요구량, 수렴, 장기 학습, 이미지 품질, ComfyUI/외부
  로더 호환성, MPS/Windows는 검증하지 않았다.
- 프리셋은 어댑터 대상을 선택한다. 기존 trainer가 MOSS의 VAE latent packing,
  Qwen text feature, noise schedule/flow target을 준비하는 기능을 추가한 것은 아니다.
