# 2026-10-08 원본 main 동기화

## 반영 범위

- 사용자 승인에 따라 로컬 `main`에 원본 `KohakuBlueleaf/LyCORIS`의
  `4a6a333819356795d22170fe661a84f16b299b6b`를 병합했다.
- 병합 전 로컬/포크 HEAD는 `fc5428515c7bae87a5e8a9ba9bc730f24374d9b2`이며,
  `codex/pre-upstream-sync-20261008` 브랜치에 보존했다.
- 원본에 새 커밋 27개, 포크에 고유 커밋 21개가 있었으며 11개 파일의
  충돌을 해결했다. 이 작업의 범위는 로컬 병합과 검증이며 원격 push는 제외한다.
- 4.0.0 패키징, Triton/TileLang 커널과 dispatch, 알고리즘 수정, exclude_name
  처리, 문서 재구성 및 CI/nightly/release 자동화를 반영했다.

## 충돌 해결과 보존한 동작

- LoKr의 full-matrix 단위 배율, 자동 full-matrix 승격, FP32 DoRA master,
  detached norm, grouped Conv, scalar resume/portable checkpoint, exact merge
  undo/conflict recovery를 유지했다. Anima의 기본 448개 대상과 패턴 override,
  Kohya optimizer grouping/freezing/apply_to 안전 규칙도 유지했다.
- `LokrModule.get_weight()`와 Linear bypass를 원본의 `kron_weight()` 및
  `kron_bypass()`에 연결했다. 그 외 LoKr 메서드는 AST 비교로 기존 동작의
  보존을 확인했다. DoRA는 원본 공통 epilogue 대신 기존 residual 계산을 사용한다.
- 함수 API의 optional `org_out`, rank/factor 검증, dtype/device 보존,
  nonsquare Tucker 전치, 그룹/비영 padding 처리를 유지했다.
- 공통 `rank_scale()`은 full-matrix에서 gamma=0을 포함해 1.0을 반환하고,
  Tucker rank는 실제 rank 축에서 계산한다. float64를 fused dtype 범위에서
  제외해 기존 정밀도를 유지한다.
- FullModule은 기존 독점 적용/rollback/hidden Parameter 안전성을 유지하면서
  scaled-add dispatch와 diff weight의 shape/device 처리를 반영했다.
- Kohya의 기존 전체 module 경로 전달에 preset exclude_name 검사를 결합했다.
  Anima 감지, regex dim/include/exclude, 조용한 unmatched 로그를 유지했다.
- requirements.txt, requirements-dev.txt, setup.py는 pyproject.toml로 이전했다.
  기존 런타임 의존성은 원본의 새 dependencies에도 포함되어 있다.
- 커널 백엔드가 없을 때 parameterized 테스트의 빈 목록이 수집 오류를 내던
  문제를 `skip_on_empty=True`로 수정했다. unit-test.py의 Kohya opt-in을 유지했다.
- 정밀 병합 테스트는 forward wrapper를 먼저 restore하여 포크의 destructive
  merge 안전 규칙을 따른다. FP64/zero-alpha full-matrix 및 grouped Tucker Conv의
  enclosing compile 전후 출력/gradient를 독립적인 PyTorch 식과 비교하는 테스트를
  추가했다. Black 형식에 맞춰 관련 파일을 정리했다.

## 검증

Python 3.12.15 / PyTorch 2.14.1+cpu의 격리된 임시 환경을 사용했다.
테스트마다 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 PYTHONPATH=.`를 지정했다.

- `python -m unittest -q test.lokr test.test_kohya_optimizer test.torch_compile
  test.precision_merge_test test.kernels.test_autograd test.kernels.test_ops`:
  **180개 중 150개 통과**, 29개 skip, 1개 expected failure. Skip은 optional Quanto/bnb
  3개와 CUDA/fused backend가 필요한 26개다. Expected failure는 부정확한
  merge/unmerge의 반복 오차를 보여 주는 기존 정밀도 테스트다.
- `python -m unittest -q test.module test.functional`: **890개 성공**.
- `python -m pytest -q test/preset_exclude_name.py`: **3개 성공**.
- `python scripts/ci/cpu_smoke.py`: **0 failures**.
- `python scripts/ci/import_check.py`: **59/59 성공**, optional backend 42개 skip.
- `python -m build --no-isolation`: 4.0.0 sdist와 wheel 생성 성공. Wheel을 임시
  환경에 설치하고 저장소 밖에서 import 경로, full-matrix DoRA forward/backward,
  scalar gradient, exact merge undo를 확인했다.
- Ruff, compileall, git diff --check 성공. Black 26.10.0의 `format_str()`를
  단일 프로세스에서 적용/비교하여 소스·테스트·unit-test.py 139개 파일 형식을
  확인했다. 다중 파일 Black CLI 검사는 대기 상태에서 중단했고 위 API 방식으로
  검증을 완료했다.
- 넓은 `test.wrapper`에는 LoRA 4개와 LoHa 6개의 실패가 남아 있다. 병합 전
  `fc54285`를 별도 임시 경로에 추출해 동일 환경에서 실행했으며 **동일한 10개
  실패 ID**가 재현되었다. 기존 비-LoKr 실패의 수정은 이번 동기화 범위에서 제외했다.

## 미검증과 다음 작업

- NVIDIA 드라이버가 응답하지 않아 실제 CUDA/Triton/TileLang 경로는 미검증이다.
  MPS/Windows와 실제 Anima 학습, Kohya/SDXL 및 Flux 전체 모델 integration도 미검증이다.
- 이후 GPU 환경에서 새 커널의 수치/gradient와 실제 사용자 학습 설정을 검증한다.
- 필요하면 기존 LoRA/LoHa wrapper 실패를 별도 작업으로 진단한다.
- 원격 반영은 별도 사용자 요청에 따라 수행한다. 검증용 임시 환경과 패키지,
  병합 전 임시 추출본은 완료 후 제거한다.
