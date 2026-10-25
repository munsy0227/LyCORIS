# 2026-10-25 LyCORIS 원본 동기화 검증

## 범위와 복구 지점

- 대상 원본: `KohakuBlueleaf/LyCORIS`의 `main`,
  `7abd5709f74f9159246ff9c6cc83bc0d5118b639` (4.1.3).
- 포크의 기존 `main`과 조회한 `origin/main`:
  `be69d8a3094e14b04dd53a2e9d7883ce1a3f7272`.
- 분기 차이: 포크 고유 27개, 원본 신규 22개 커밋.
  원본 변경은 100개 파일, 13,781줄 추가와 451줄 삭제.
- 백업: `codex/pre-upstream-sync-20261025`가 기존 `main`을 가리킨다.
- 검증 브랜치: `codex/upstream-sync-20261025`.
  기존 포크 커밋을 보존하는 merge이며 reset/rebase/force push는 하지 않았다.
- 기존 체크아웃 `/home/munsy0227/LyCORIS`의
  `codex/moss-sd-scripts-integration` (`169dc642aa662f85d8741ba18651907462abb753`)과
  미추적 worklog 2개 및 `artifacts/`는 보존했다.
- 현재 결과는 별도 로컬 검증 브랜치에만 있다. 원격 `main` 반영은 사용자 확인 후 진행한다.

## 반영 내용과 충돌 해결

- 원본의 장치별 backend 선택/강등, 비용 모델과 튜너, timing 스위치,
  RNG 상태 보존, custom op 등록, LoKR token-major GEMM과 4.1.3 버전을 반영했다.
- `autograd/lokr.py`: 새 base-output 합산을 수용하면서 Tucker의 rank 축과
  full-matrix 단위 배율 및 `gamma=0` 동작을 보존했다.
- `gradbuf.py`: scratch와 tuning twin의 256바이트 FP32 offset 정렬을 보존했다.
  저정밀도 반환 gradient는 원본의 독립 저장소 방식으로 바꿔 custom op의
  alias 제약을 만족시켰다. CPU 회귀 테스트 2개를 추가했다.
- `select.py`: 원본의 `device_of()`와 backend 흐름을 수용하면서
  FP16/BF16/FP32만 fused 경로에 넣는 기존 제한을 보존했다. FP64는 eager로 간다.
- `triton/lora/bypass.py`: 원본의 분기별 서로 다른 SSA 변수 이름을 사용했다.
- GPU 검증에서 새 TileLang OFT `solve.py`/`rotate.py`의 텐서 인자 `res`가
  NVRTC 래퍼의 CUDA 반환 상태 변수와 충돌하는 오류를 재현했다.
  여섯 TIR 함수의 인자를 `rescale`로 바꾸고 전방/역전파를 다시 검증했다.
- 기존 LoKR/DoRA 수학, magnitude FP32 master, checkpoint/resume,
  grouped convolution, reversible merge ledger 및 Anima/MOSS 범위는 유지했다.

## 최종 검증

환경: Linux, Python 3.12.15, PyTorch 2.14.1+cu130, RTX 4070,
Triton 3.8.0, TileLang 0.1.15 (`nvrtc`). 기존 ComfyUI 환경은 변경하지 않고
`/tmp/lycoris-sync-py312`에서 기존 PyTorch를 참조했다.
CPU thread 수는 각 1로 제한하고 기본 tuning은 껐다.
튜닝/RNG 테스트는 테스트 내부에서 timing을 켜 해당 경로도 검증한다.

최종 소스를 새 프로세스에서 실행한 결과:

| 묶음 | 통과 | 건너뜀 | 추가 unittest subtest 통과 |
| --- | ---: | ---: | ---: |
| 전체 kernel + LoKR + 전용 CUDA + Kohya optimizer + compile + MOSS preset | 414 | 6 | 181 |
| 기존 module/functional/wrapper LoKR 조합, Triton | 435 | 0 | 4 |
| 위 LoKR 조합 + focused LoKR + 전용 CUDA, TileLang | 570 | 3 | 158 |

백엔드별 반복을 포함한 합계는 1,419개 통과, 9개 건너뜀이다.
첫 묶음은 다음 파일을 사용했다:

```text
test/kernels/
test/test_lokr.py
test/test_lokr_cuda.py
test/test_kohya_optimizer.py
test/torch_compile.py
test/test_moss.py
```

435개 조합은 기존 parameterized 테스트의 인자가 `LokrModule` 또는
`functional.lokr`인 항목과 이름에 `lokr`가 있는 회귀/랩퍼 테스트를 선택했다.
실행 환경 변수는 다음과 같다:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TILELANG_EXECUTION_BACKEND=nvrtc
LYCORIS_KERNEL_TUNE=0
XDG_CACHE_HOME=/tmp/lycoris-sync-cache
TRITON_CACHE_DIR=/tmp/lycoris-sync-cache/triton
```

- 전체 `lycoris/ scripts/ test/` Ruff 및 Black 검사 통과.
- Python 185개 파일 AST 검사와 `git diff --check` 통과.
- 최종 `lycoris_lora-4.1.3-py3-none-any.whl` 빌드 및 새 커널 포함 여부 확인.
- wheel을 임시 환경에 설치하고 `python -I`로 소스 경로를 제외하여 검증:
  두 백엔드의 CUDA BF16 LoKR bypass/gradient, FP64 eager 폴백,
  full-matrix DoRA backward, checkpoint 복원, bitwise 정확한 merge undo 통과.
- 로그/XML/실행 보조 스크립트:
  기존 체크아웃의 `.internal/sync-20261025/` (Git 제외).

## 검증 한계

- 각 LoKR focused 묶음에서 optional bitsandbytes/Quanto 테스트 3개를
  의존성 미설치로 건너뛰었다. 첫 묶음에서는 `LYCORIS_MOSS_MODEL_PATH`가
  없어 실제 MOSS 모델 관련 3개 테스트도 건너뛰었다.
- Windows, MPS, 다른 GPU에서의 실행, 실제 대형 모델 학습 및
  장시간 성능/메모리 측정은 하지 않았다.
- 이 검증은 GitHub hosted CI 실행 결과를 대신하지 않는다.
