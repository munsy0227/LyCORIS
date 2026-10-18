# 공식 MOSS trainer 브랜치 검토

후속: 검토 후 승인된 stage 3 통합은 [구현 기록](2026-10-18-official-moss-trainer-integration.md)에 정리했다. 아래 문제는 검토한 고정 upstream 커밋의 상태이며, 로컬 구현에서는 보완했다.

- 확인 시각: 2026-10-18 12:14 KST
- 요청: 새 `feat-moss-image-train-network` 브랜치를 현재 통합 작업과 비교
- 범위: 읽기 전용 소스 검토, 메타 모델 대상 비교, 독립된 메서드의 제어 흐름 재현. 이 기록 이외의 파일 변경, merge, commit, push 없음.

## 결론

공식 trainer를 다음 통합의 기반으로 삼을 가치가 있다. 현재 자체 trainer와 겹치는 부분이 많고 timestep sampling, compile, offload, metadata, 추론 sampler 공통화가 추가됐다. 다만 **이 커밋을 그대로 대체 적용하는 것은 권하지 않는다.** 검증 배치 처리에 재현되는 오류가 있고, 기존 캐시/메모리/프리셋/체크포인트 호환 기능도 보존해야 한다.

권장 구조는 공식 `moss_image_train_network.py`를 중심에 두고, 기존 `moss_train_network.py`를 기존 CLI/TOML 및 LyCORIS 동작을 보존하는 호환 진입점으로 연결하는 방식이다. 실제 전환 시에는 아래 문제를 먼저 보완하고 회귀 검사를 해야 한다.

## 확인한 upstream 상태

| 항목 | 결과 |
| --- | --- |
| 브랜치 HEAD | `cb5c24fc724f758fcb1e63969872f2f561ac1396` |
| 커밋 시각 | 2026-10-18 11:30:27 KST |
| 현재 공식 `dev` | `3e84e58f647a4246892d7857186715c8e0b6c408` |
| 기존 검토 기준과 차이 | 해당 `dev`에서 1 commit ahead, 0 behind |
| PR | [#2460 — MOSS-Image LoRA training, stage 3](https://github.com/kohya-ss/sd-scripts/pull/2460) |
| PR 상태 | open, draft 아님, 아직 merge 안 됨. 확인 시 review/comment 없음 |
| 변경 크기 | 13개 파일, +1,964 / -198 |

[고정 커밋](https://github.com/kohya-ss/sd-scripts/commit/cb5c24fc724f758fcb1e63969872f2f561ac1396), [이전 기준과 전체 비교](https://github.com/kohya-ss/sd-scripts/compare/3e84e58f647a4246892d7857186715c8e0b6c408...cb5c24fc724f758fcb1e63969872f2f561ac1396).

## 현재 구현에 도움이 되는 변경

| 추가 내용 | 평가 / 통합 방향 |
| --- | --- |
| 공식 `MossImageNetworkTrainer` | 모델·캐시를 넘어 공식 학습 진입점이 추가됐다. 장기적으로 이 인터페이스를 기반으로 유지하는 것이 좋다. |
| 공용 timestep sampling / weighting | 현재 `flow_batch`의 이산 uniform 선택보다 설정 범위가 넓다. sigma, uniform, sigmoid, shift, flux_shift와 sample별 offset을 연결한다. 기존 설정의 의미는 보존하고 새 분포를 명시적으로 선택하게 하는 것이 좋다. |
| timestep / sigma FP32 | BF16에서 timestep 정밀도가 줄어드는 문제를 피한다. 현재 로컬 `flow_batch`도 FP32를 사용하므로 이 부분은 이미 같은 방향이다. |
| 공통 `generate_latents` | 학습 중 sampling과 minimal inference가 같은 sampler를 사용하도록 구성한다. 로컬의 별도 sampling 루프를 줄일 수 있다. |
| per-block `torch.compile`, CUDA 옵션, unsloth offload | 현재 trainer에 없는 선택지가 추가됐다. block swap / CPU offload / fullgraph 등 조합 제약도 함께 가져와야 한다. |
| `--show_timesteps` | 모델을 학습하기 전에 분포를 확인할 수 있어 유용하다. |
| `networks.lora_generic` 및 MOSS `ArchConfig` | LoRA와 기존 `networks.loha` / `networks.lokr`가 공용 대상 탐색을 사용한다. 기존 `lycoris.kohya`를 유지하면서 별도 선택지로 제공할 수 있다. |
| Qwen TE LoRA 및 TE weight merge | upstream은 학습 및 추론 시 TE adapter merge까지 추가했다. 현재 작업은 TE frozen을 기본으로 유지하고, TE 학습은 별도 opt-in 범위로 검증하는 것이 좋다. |
| SAI model spec | `moss-image-base` architecture 식별을 공용 metadata에 추가한다. 채택할 만하다. |

모델/attention/block swap 구현과 text/latent cache strategy 자체는 이번 13개 변경에 포함되지 않는다. 기존 dev 통합의 공식 모델과 캐시 기반을 다시 바꾸는 커밋은 아니다. 문서와 int8 ConvRot은 PR 설명에서도 후속 작업으로 남아 있다.

공식 PR이 권장하는 pretraining 분포 재현 설정은 다음과 같다. 이는 PR 작성자의 권고이며, 이번 검토에서 실제 pretrained 학습 분포를 별도 검증한 것은 아니다.

```text
--timestep_sampling sigma --weighting_scheme logit_normal \
--logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0
```

출처: [trainer 학습/메모리 옵션](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/moss_image_train_network.py#L47-L170), [sampling/compile 인자](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/library/moss_image_train_utils.py#L59-L112), [공용 sampler](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/library/moss_image_train_utils.py#L208-L256).

## 그대로 가져오기 전에 보완할 문제

### 1. 캐시를 사용하는 validation의 dropout 및 배치 변형

공식 `process_batch`는 `is_train=False`인 경우에도 `drop_cached_text_encoder_outputs`를 호출한다. 이후 원래 다섯 필드인 `batch["text_encoder_outputs_list"]`를 `[선택된 embedding, mask, dropout rate]` 세 필드로 덮어쓴다.

공용 validation 루프는 한 배치를 여러 `validation_timesteps`에서 재사용한다. 두 번째 호출은 세 필드에서 마지막 rate를 제외한 두 필드만 전달한다. 실제 strategy 함수는 conditional/mask와 unconditional/mask 네 필드를 요구하므로 `TypeError`가 발생한다. dropout rate가 0이어도 인자 부족은 동일하다.

최소 재현은 upstream AST에서 실제 메서드 본문을 추출해 가짜 base/strategy에 연결했다. 텐서 연산이나 모델 학습을 모사한 결과가 아니라, 해당 메서드의 분기 및 인자 전달을 실행한 결과다.

```text
is_train=False: dropout 호출 1회, 원본 batch의 필드 수 5 → 3
같은 batch의 두 번째 호출:
TypeError: ... missing 2 required positional arguments: 'uncond_embeds' and 'uncond_mask'
```

통합 시 학습에서만 dropout을 적용하고, 원본 cache tuple/batch를 훼손하지 않아야 한다. 현재 로컬 trainer는 `get_noise_pred_and_target`에서 `is_train`을 확인하고 원본 다섯 필드를 유지하므로 이 처리를 보존해야 한다.

출처: [공식 process_batch](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/moss_image_train_network.py#L351-L397), [strategy 함수 계약](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/library/strategy_moss_image.py#L87-L113), [배치를 재사용하는 validation 루프](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/train_network.py#L820-L846).

### 2. TE는 CPU로 이동하지만 sample prompt 임베딩은 GPU에 남음

공식 캐시 훅은 TE를 accelerator로 옮겨 dataset 및 sample prompt를 인코딩한 뒤 TE 자체는 CPU로 옮긴다. 그러나 `sample_prompts_te_outputs`에는 인코딩 결과를 `.cpu()` 없이 저장한다. encoding/padding 함수는 입력 TE와 같은 device에 결과를 만들기 때문에 이 텐서들은 DiT가 로드될 때도 accelerator 메모리를 점유한다.

가짜 device tensor로 실제 캐시 훅의 제어 흐름을 재현했을 때도 `TE=cpu`, `stored prompt tensors=cuda:0`이었다. **실제 CUDA 메모리 사용량을 측정한 것은 아니다.** 영향은 prompt 수와 길이에 따라 달라진다.

현재 로컬의 sample/unconditional 임베딩 CPU 보관과, 모든 dataset 처리 후 DiT를 배치하기 직전 한 번 TE 참조를 정리하는 순서를 유지해야 한다. 공식 훅은 train/validation dataset마다 prompt 인코딩 및 CPU↔GPU 이동을 반복하므로 단순 교체하면 현재 정리 순서도 달라진다.

출처: [공식 TE/cache 훅](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/moss_image_train_network.py#L211-L260), [encoding/padding의 device 유지](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/library/moss_image_utils.py#L260-L286).

## LyCORIS 범위와 호환성

공식 모델이 이번 커밋에서 바뀌지 않았음을 확인하고, 현재 공식 모델을 `torch.device("meta")`에서 생성했다. upstream `ArchConfig`의 class/Linear 탐색 규칙과 현재 LyCORIS 정규식 프리셋을 같은 `named_modules()` 목록에 적용했다. 가중치 로드나 forward는 실행하지 않았다.

| 선택 | DiT 대상 수 | 비교 |
| --- | ---: | --- |
| 공식 기본값 | 252 | 현재 `moss`와 대상 집합이 정확히 동일 |
| 공식 `train_text_embedder=true` | 263 | 기본 252 + text projection/refiner 11 |
| 현재 LyCORIS `moss` | 252 | 공식 기본값과 대칭차집합 없음 |
| 현재 LyCORIS `moss-full` | 269 | 공식 text embedder 포함 범위보다 아래 6개가 더 많음 |

추가 6개는 `time_embedding.0`, `time_embedding.2`, `video_embedder`, `shared_block_modulation`, `final_linear_video`, `video_head_modulation`이다. 따라서 **`train_text_embedder=true`를 `moss-full`의 동의어로 취급하면 안 된다.**

이 결과는 대상 경로 비교다. 서로 다른 network 구현의 생성 순서, adapter tensor 형태, optimizer state 호환까지 증명하지 않는다.

공식 `networks.lokr`도 full matrix 경로를 갖지만 DoRA 구현은 없다. 그 지원이 기존 LyCORIS full-matrix LoKR+DoRA의 동등성을 의미하지 않는다. 기존 `lycoris.kohya`의 수학, preset, native/portable key와 재개 경로를 그대로 유지하면서 새 trainer와 연결해야 한다.

출처: [공식 MOSS ArchConfig](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/networks/network_base.py#L56-L68), [공식 대상/alias 검사](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/tests/test_lora_generic.py#L12-L16), [공식 native LoKr](https://github.com/kohya-ss/sd-scripts/blob/cb5c24fc724f758fcb1e63969872f2f561ac1396/networks/lokr.py).

## 현재 작업과 통합 경계

공식 변경 13개 파일 중 기존 파일 9개를 이전 공식 `dev` 원문과 현재 작업본으로 비교했다. 7개는 현재 작업본이 이전 공식 기준과 동일하고, 로컬 확장이 겹치는 파일은 다음 둘이다.

- `library/moss_image_utils.py`: root/component 자동 탐색, 명시 경로 우선순위, component/cache 식별을 보존하며 TE merge 인자를 추가해야 한다.
- `library/moss_image_train_utils.py`: 로컬 caption/layout 제약, 기존 schedule 및 flow helper와 공식 training/sampling helper를 정리해야 한다.

파일이 겹치지 않는 기능도 동작 연결이 필요하다. 특히 다음 기능은 upstream trainer를 적용하는 것만으로 유지되지 않는다.

- 기존 `moss_train_network.py` 명령과 TOML, 모델 root 및 로컬 tokenizer/offline 사용
- TXT/JSON/레이아웃 캡션 전처리와 캐시 도구의 동일한 처리
- v2 cache fingerprint/무효화 및 `.legacy-v1.bak` 보존
- 학습 단계에서만 caption dropout 적용
- 모든 dataset/sample prompt 인코딩 후 TE와 보관 임베딩 정리
- 기존 native checkpoint / portable adapter, optimizer/scheduler/global step 재개
- LyCORIS `moss`/`moss-full`, LoRA/LoKR/full-matrix LoKR+DoRA 및 ComfyUI 검사

확인 당시 sd-scripts는 `codex/sd-scripts-moss-dev-integration`의 기존 미커밋 merge 상태를 그대로 유지했다. HEAD는 `2de3c83...`, MERGE_HEAD는 `3e84e58...`, staged 파일 51개, unstaged 변경과 unresolved conflict는 없었다. 이번 브랜치 검토에서 이 상태를 변경하지 않았다.

## 검증 범위와 후속 우선순위

이번 확인에서 실행한 것은 GitHub 공식 API/고정 소스 비교, 메타 모델의 정확한 대상 집합 비교, 두 upstream 메서드의 가짜 의존성 제어 흐름 재현이다. 새 브랜치 전체 테스트, 실모델/CUDA 학습, TE LoRA, compile/offload 성능, 새 generic adapter의 ComfyUI 적용은 실행하지 않았다. PR 작성자의 RTX PRO 6000 실가중치 검증은 PR에 기재된 upstream 결과이며 이번 로컬 검증 결과가 아니다.

후속 구현 시 우선순위:

1. 공식 trainer를 기반으로 기존 명령의 호환 계층을 만들고, 위 validation 및 prompt device 문제를 보완한다.
2. 기존 LyCORIS/캡션/캐시/재개 기능을 연결하고 이전 회귀 검사에 복수 validation timestep 검사를 추가한다.
3. timestep 분포와 공통 sampler, metadata를 연결하되 기존 설정에서 바뀌는 의미를 명시한다.
4. compile/unsloth 및 TE LoRA는 각각 opt-in으로 검증한다. quantization/edit 지원은 계속 후속 범위로 둔다.
