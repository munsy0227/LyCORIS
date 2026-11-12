# MOSS·Anima 작업의 main 통합

## 범위와 복구

- 사용자 요청으로 sd-scripts와 LyCORIS의 MOSS·Anima 작업을 각 main에 통합한다.
- LyCORIS 통합은 main `deeb89ebf9e2eed94730aef80ec39a507f2696ae`에서 만든 `codex/moss-anima-main-20261112`의 별도 worktree에서 수행했다.
- 병합 전 refs는 `/home/munsy0227/LyCORIS/.internal/backups/main-integration-20261112-101225`의 bundle에 별도로 보존했다.
- 원래 checkout의 미추적 작업 기록 두 개와 `artifacts/`는 변경하지 않았다.

## 병합

- `codex/moss-training-speed` (`6019ca061ecebe2fff0374cf50b86329677cd025`), 이어 `codex/moss-sd-scripts-integration` (`169dc642aa662f85d8741ba18651907462abb753`)을 이력을 보존하는 일반 merge로 통합했다.
- 두 병합 모두 충돌 없이 완료했다. 양쪽이 수정한 MOSS 사용 문서도 자동 결합됐다.
- 과거 Anima 작업 `origin/agent/lokr-full-matrix-dora` (`75f35c97f2cee58a564553fdf634d14bb2b74daa`)는 이미 main의 조상이므로 중복 병합하지 않았다.
- 최신으로 확인한 upstream main `7abd570`도 통합 결과의 조상이다.
- 별도 코어 수정 없이 기존 LoKR DoRA 최적화, 공식 sd-scripts 연동 회귀 검사, 문서와 작업 기록을 함께 보존했다.

## 검증

- main, 두 MOSS 브랜치, 과거 Anima 브랜치, upstream main이 모두 통합 결과의 조상임을 확인했다.
- `git diff --check main..HEAD`를 통과했다.
- 변경된 Python 파일 5개의 AST 파싱을 통과했다.
- 문서에 MOSS 252개, 공식 text-embedder 선택 263개, `moss-full` 269개 및 compile 사용 설명이 함께 남아 있음을 확인했다.
- Python 3.12, PyTorch 2.13.0+cu130의 별도 `/tmp/codex-moss-anima-env`에서 CPU 스레드 1개와 Hugging Face offline 설정으로 검사했다. 기존 설치 환경은 변경하지 않았다.
- CPU에서 `test.lokr`, `test.test_kohya_optimizer`, `test.torch_compile`, `test.test_moss`, `test.test_moss_sd_scripts`: **163 tests, 158 passed, 5 skipped**, 11.249초. skip은 optimum-quanto 미설치 2개와 `LYCORIS_MOSS_MODEL_PATH` 미지정에 따른 로컬 pretrained 검사 3개다.
- RTX 4070에서 `test.test_lokr_cuda`, `test.test_moss_sd_scripts`: **6 passed, 0 skipped**, 51.011초. FP32/FP16/BF16 DoRA 수학·gradient, compiled checkpoint, 정확한 merge/undo, mock Anima 448개 BF16 optimizer step 및 공식 sd-scripts MOSS 양 factory 연동과 252/269 범위를 확인했다.
- 두 저장소의 실제 통합 worktree를 `PYTHONPATH`에 함께 지정했다. CPU에서는 `CUDA_VISIBLE_DEVICES=''`로 GPU 실행을 차단했고 CUDA 검사는 별도로 실행했다.
- 로그: `/tmp/lycoris-moss-anima-cpu-20261112.log`, `/tmp/lycoris-moss-anima-cuda-20261112.log`.
- 이 단계에서 main ref와 원격 브랜치는 변경하지 않았다.

## 보존 기준과 제한

- full-matrix LoKR unit scaling, FP32 DoRA magnitude와 detached norm, scalar/checkpoint 호환, 정확한 merge/undo 및 Anima 448개 범위를 유지한다.
- 실제 pretrained 전체 학습, 수렴·화질·속도, Windows/MPS는 이번 통합의 검증 범위에 포함하지 않는다.
