# sd-scripts MOSS 통합 문서·검사 게시

사용자의 “그것도 반영해서 이제 커밋 푸시하자” 요청에 따라 이 작업에서 만든 문서, 회귀검사 및 검토 기록을 `codex/moss-sd-scripts-integration` 브랜치에 묶는다. LyCORIS core와 main ref는 변경하지 않는다. 사용자 `artifacts/` 및 `2026-10-18-moss-autostop-script.md`는 작업 범위에서 제외한다.

## sd-scripts 결과

- `be216b479e2c10460aa7a38f0323a920f25eee5e`: 기존 공식 dev 모델/캐시 및 stage 3 trainer 통합, LyCORIS·캡션·캐시·재개 호환.
- `ebec22a60493471c5f4fd5a6ecb185b409ec9a76`: 공식 dev `6a3290182c786541ccf292106782b51568e46da5` 정상 병합. sample tqdm 복구와 debug dataset의 None 처리 수정 두 개를 포함한다.
- 두 번째 커밋의 부모는 첫 통합 커밋 및 공식 dev이며 이력 재작성은 없다. 기존 validation/TE 메모리/CPU offload 보완을 보존했다.
- 게시 대상: `munsy0227/sd-scripts`의 `codex/sd-scripts-moss-dev-integration`.

## LyCORIS 범위

`docs/usage/moss-image.md`는 공식 trainer와 구형 명령의 관계, 252/263/269개 대상 범위 및 현재 dev 기준을 설명한다. `test/test_moss_sd_scripts.py`는 선택적으로 설치한 공식 sd-scripts 모델에서 두 프리셋과 LoRA/LoKR/LoKR+DoRA의 adapter lifecycle을 검사한다. core의 수학이나 기본 설치 의존성을 바꾸지 않는다.

검토/구현 기록 4개와 이 게시 기록을 함께 보존한다. 이전 기록의 미커밋 상태와 미검증 항목은 해당 단계의 시점 기록이다. 이후 commit/push 승인은 이 기록에서 구분한다.

## 검증

이전 구현 단계의 CPU/CUDA BF16 adapter matrix, trainer save/resume, 이전 checkpoint step 2→3, ComfyUI 검사 결과는 기존 구현 기록에 있다. 이번 두 후속 수정에는 다음 추가 검사를 수행했다.

- 실제 `debug_dataset`: conditioning/masked가 둘 다 None인 경우, masked fallback, conditioning 우선의 3개 경우 정상.
- 실제 CPU/CUDA sampling→학습 복귀: 2 passed. tqdm 출력도 확인했다.
- AST/whitespace 및 변경 helper Ruff 정상. 공용 dataset의 기존 Ruff 진단은 증가하지 않았다.

전체 pretrained 품질·성능·장기 실행·Windows/MPS 등 기존 미검증 범위는 유지한다. 기존 `.venv`, 이전 native fixture, `.internal/backups/` 복구 자료를 보존하고 이번 임시 검증 환경만 정리한다. 게시 후 로컬 HEAD와 원격 branch ref를 대조한다.
