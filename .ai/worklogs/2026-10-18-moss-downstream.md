# MOSS Image downstream 학습·ComfyUI 연결

- 사용자가 승인한 계획에 따라 sd-scripts의 MOSS trainer/TXT·JSON captions와 ComfyUI portable adapter 연결을 구현했다. LyCORIS 연산 코드는 변경하지 않았다.
- 기존 sd-scripts Anima BF16 커밋은 백업 브랜치와 검증한 bundle로 보존한 뒤 로컬 main에서 제거했다. bundle은 `.internal/backups/sd-scripts-pre-moss-20261018.bundle`이고 SHA256은 `b8333ac6b8f1b9a8c0c78bb62c47c78c83f73432364a588c7bc858d73e6159ad`다.
- 상세 변경·검증·제한은 `/home/munsy0227/sd-scripts/.ai/worklogs/2026-10-18-moss-image.md`에 기록했다. `docs/usage/moss-image.md`에 downstream 연결을 추가했다.
- 테스트는 표준 LoRA, full-matrix LoKR, full-matrix LoKR+DoRA의 두 프리셋과 CPU FP32/CUDA BF16을 포함한다. ComfyUI는 generic lycoris_와 Kohya lora_unet_ prefix, 252/269개 대상 및 QKV slice를 검증했다.
- 최종 선택149개 통과. native 재개 시 공통 sd-scripts 루프가 global step을0으로 되돌리던 문제도 수정하고 실제 step1→2 진행을 확인했다.
- 추가 진단에서 기존 generic LoRA+DoRA merge의 multiplier 0/0.5 동작 차이를 확인했다. 이 비-LoKR 기존 문제는 이번 범위에서 수정하지 않았다.
- 전체 pretrained 학습/이미지 생성/화질·VRAM·Windows/MPS는 미검증이다. 새 커밋·푸시·원격 이력 변경은 하지 않았다.
- 이후 사용자가 커밋·푸시를 요청했다. 이 저장소에서는 downstream 문서와 본 작업 기록만 커밋하고 main에 fast-forward로 게시한다. sd-scripts 이력 교체와 ComfyUI 게시 대상은 각각 별도로 확인한다.
