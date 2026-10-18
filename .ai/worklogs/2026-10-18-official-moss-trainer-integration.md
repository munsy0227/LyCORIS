# 공식 MOSS stage 3 trainer 연결

후속: 최신 dev 후속 수정과 커밋/푸시 작업은 [게시 기록](2026-10-18-moss-dev-publication.md)에 정리했다.

새 브랜치 검토 후 사용자의 “그럼 해보자” 승인으로 공식 `cb5c24fc724f758fcb1e63969872f2f561ac1396` 변경을 기존 sd-scripts 통합 작업에 반영했다.

## 결과

- 공식 `moss_image_train_network.py`를 기반으로 기존 `moss_train_network.py` 명령을 호환 subclass로 유지했다.
- LyCORIS 수학, `moss` 252개/`moss-full` 269개, TXT/JSON/layout, 모델 경로와 캐시·체크포인트 호환을 보존했다.
- 공식 generic text-embedder 선택은 263개다. 이를 `moss-full`로 치환하지 않는다.
- validation의 원본 다섯 필드 cache를 유지하고 학습에서만 dropout을 적용한다.
- frozen TE/sample prompt를 모두 사전 계산한 뒤 CPU tensor를 보관하고, DiT 로드 전에 TE를 한 번 해제한다.
- Qwen은 BF16 weight/FP32 rotary buffer를 사용하며 DiT autocast와 분리한다. 이전 encoding fingerprint는 cache 백업/재생성을 유발한다.
- timestep/compile/unsloth/공통 sampler/metadata를 연결했다. 기존 명령의 timestep 기본값은 `legacy`, 공식 명령은 `sigmoid`다.
- 실제 CUDA 검사에서 발견한 CPU checkpoint offload의 device 오류를 수정했다. 저장 입력만 CPU로 옮기고 live output은 GPU에 둔다.

## 검증

RTX 4070과 PyTorch 2.13.0+cu130, transformers 5.17.0, diffusers 0.40.0의 별도 임시 환경을 사용했다. 기존 `.venv`는 수정하지 않았다.

- 공식/기존 trainer와 generic LoRA 선택 51 passed, 새 통합 회귀 10 passed.
- 두 진입점의 별도 프로세스 CUDA BF16 학습·저장·재개 2 passed.
- 실제 축소 Qwen 로더 FP32/BF16 CPU 및 BF16 CUDA 3 passed. 최종 encoding 영향 범위 17 passed.
- 공식 모델/캐시/캡션/LR/offload 선택 96 passed; opt-in ComfyUI 및 XPU/MPS가 skip된 뒤 실제 ComfyUI 경로로 49 passed를 별도 확인했다.
- LyCORIS MOSS/Kohya unittest 22 tests OK, optional 3 skipped. sd-scripts 모델 matrix의 24 subtests 포함.
- 이전 trainer 상태에서 step 2→3, 이전 portable adapter 로드, 복수 validation timestep, sampling 실패 복귀, CPU `eager` compile과 CUDA CPU/unsloth offload gradient 비교를 확인했다.

이 숫자는 중복 없는 전체 테스트 수로 합산하지 않는다. pretrained 전체 학습·품질, CUDA Inductor 성능, real 9B TE LoRA, TE adapter ComfyUI, Windows/MPS 및 새 양자화/edit 지원은 미검증 또는 후속이다.

## Git 및 복구

sd-scripts의 기존 미커밋 dev merge를 유지한 채 stage 3 diff를 추가했다. HEAD `2de3c83...`와 MERGE_HEAD `3e84e58...`는 유지했다. commit/push/main 반영은 수행하지 않았다.

원래 staged 작업의 복구 패치는 `.internal/backups/sd-scripts-pre-stage3-20261018.patch` 및 같은 basename의 JSON에 있다. 기존 native fixture, 사용자 `artifacts/`, 별도 autostop worklog를 보존했다.

최종 sd-scripts index는 삭제를 포함한 64개 경로이며 unresolved/unstaged/untracked가 없다. Python 50개 AST, MOSS Ruff F/format과 양 저장소 whitespace 검사를 통과했다. 임시 검증 환경과 다운로드 cache를 삭제했고 LyCORIS core는 변경하지 않았다.

상세 구현·명령·제한은 `/home/munsy0227/sd-scripts/.ai/worklogs/2026-10-18-official-moss-trainer-integration.md`에 기록했다. 사용 가이드는 양 저장소의 MOSS 문서와 sd-scripts `docs/moss_image_train_network.md`를 갱신했다.
