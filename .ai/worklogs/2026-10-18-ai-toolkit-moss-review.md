# ai-toolkit MOSS Image 구현 비교 검토

> 아래는 구현 전 조사다. 이후 공식 sd-scripts dev를 기준으로 재구성했으며,
> 최신 구현·검증 상태는 [후속 작업 기록](2026-10-18-official-moss-dev-integration.md)을 따른다.

검토일: 2026-10-18 KST. 대상은 현재 LyCORIS와 로컬 sd-scripts이다. 이번 작업은 조사와 비교이며, 실행 코드·의존성·Git 이력은 변경하지 않았다.

## 판단

가장 먼저 도입할 만한 것은 **sd-scripts의 Qwen 텍스트 전용 로딩**이다. 다음은 **Transformer·텍스트 인코더·VAE 경로 분리와 ComfyUI 가중치 입력**, 그다음은 **선택적 양자화 학습**이다. LyCORIS의 LoKR/DoRA 수학이나 기존 MOSS 프리셋을 교체할 근거는 찾지 못했다.

| 순서 | 적용 대상 | 제안 | 기대 효과와 범위 |
| --- | --- | --- | --- |
| 1 | sd-scripts | Qwen3.5의 language tower만 로드 | 사용하지 않는 vision tower의 로딩·보관을 줄임. 주로 RAM과 텍스트 캐시 단계의 메모리 개선 |
| 2 | sd-scripts | 세 컴포넌트의 독립 경로와 실수형 ComfyUI 단일 파일 지원 | 기존 모델 파일 재사용, Base/SFT 선택 편의, 파일 중복 감소 |
| 3 | sd-scripts + 필요한 LyCORIS 경계 | 텍스트 인코더 양자화부터 도입하고 DiT 양자화로 확장 | 캐시 단계와 본 학습의 메모리 절약. 출력 오차·gradient·DoRA 호환성 검증 필요 |
| 4 | 양쪽의 호환성 검사 | ai-toolkit과의 어댑터 저장·불러오기 검증 | QKV 구조와 LoRA/LoKR 형식 차이로 생길 혼동 방지 |

현재 구현에도 텍스트 캐시, 캐시 후 Qwen 해제, gradient checkpointing, block swapping이 있다. 따라서 offload 체계를 새로 가져오는 것보다 누락된 로더·양자화 경계를 보완하는 편이 범위와 효과가 명확하다.

## 비교 기준

- ai-toolkit: `9a1828f20131f3dc4fac8bab75bcfdf53df96398`.
- LyCORIS: `be69d8a3094e14b04dd53a2e9d7883ce1a3f7272`.
- sd-scripts: `2de3c83f925a06f5f14f4939633bff1be807ec18`.
- OpenMOSS 공식 GitHub: `c1355e6b1986dfe8fba3c0455783a05899212342`.
- 공식 MOSS-Image-Base 모델: `1ceede80badddd6a7812069619edaa5633587192`.

LyCORIS는 기존 `artifacts/` 미추적 폴더가 있었고 sd-scripts는 깨끗했다. 기존 파일은 건드리지 않았다. 과거 작업 기록의 승인·게시 지시는 이번 요청의 변경 승인으로 사용하지 않았다.

ai-toolkit에는 MOSS의 Base/SFT와 reference image를 사용하는 edit 변형까지 등록되어 있다. edit는 코드 자체에서 원래 모델이 T2I이며 reference conditioning을 확장한 것이라고 설명한다. 공식 모델의 모든 기능이 같은 방식으로 검증되었다는 의미는 아니다. [모델 등록](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/toolkit/models/registry.py#L59-L62), [MOSS 연결 코드](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/extensions_built_in/diffusion_models/moss_image/moss_image.py#L138-L175).

## 1. Qwen 텍스트 전용 로딩: 우선 적용 권장

현재 [moss_train_network.py](/home/munsy0227/sd-scripts/moss_train_network.py:98)는 `AutoModel.from_pretrained(root / "text_encoder")`로 모델을 읽는다. 공식 config의 architecture는 `Qwen3_5Model`, model type은 `qwen3_5`이다. 공식 체크포인트 index의 759개 키 중 333개는 `visual.*`, 426개는 `language_model.*`이다. 현재 TXT·JSON·레이아웃 캡션 학습 경로는 이미지 입력을 Qwen에 전달하지 않는다. [공식 config](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/1ceede80badddd6a7812069619edaa5633587192/text_encoder/config.json), [공식 checkpoint index](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/1ceede80badddd6a7812069619edaa5633587192/text_encoder/model.safetensors.index.json).

ai-toolkit은 T2I에서 `Qwen3_5TextModel`을 사용한다. 전체 config에서 `text_config`를 선택하고 checkpoint key prefix를 변환하며, vision tower 등 사용하지 않는 부분을 제외한다. edit 경로에서만 vision tower를 유지한다. 이 분리는 현재 sd-scripts의 텍스트 전용 학습에도 적용할 가치가 있다. [Qwen 로더](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/toolkit/models/v2/text_encoders/qwen35.py).

구현 시에는 단순히 클래스 이름만 바꾸면 안 된다. `language_model.` prefix 변환, 필요한 key의 누락 검사, 같은 tokenizer·chat template·최종 hidden state 사용을 함께 처리해야 한다. 기존 full model과 text-only model의 임베딩을 같은 입력으로 비교하고, CPU 및 BF16 경로의 오차를 확인하는 것이 완료 조건이다.

현재 기본값은 Qwen을 CPU에서 캐시하고 DiT를 로드하기 전에 해제하는 방식이다. 따라서 이 변경으로 **학습 중 상시 VRAM이 크게 감소한다고 단정할 수 없다**. 우선 기대할 효과는 모델 로딩과 텍스트 사전 계산의 RAM·선택적 GPU 캐시 단계 메모리 감소이다. 실제 절감량은 측정하지 않았다.

## 2. 컴포넌트 경로 분리와 ComfyUI 파일 로딩: 적용 권장

현재 trainer는 하나의 루트에 있는 `text_encoder/`, `vae/`, `transformer/`를 직접 사용한다. 반면 ai-toolkit은 별도 TE 경로, 별도 VAE 경로, Transformer 단일 `.safetensors`, 공식 폴더와 ComfyUI repack을 처리한다. [현재 로딩](/home/munsy0227/sd-scripts/moss_train_network.py:98), [현재 Transformer 로딩](/home/munsy0227/sd-scripts/moss_train_network.py:130), [ai-toolkit 로딩](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/extensions_built_in/diffusion_models/moss_image/moss_image.py#L189-L327).

권장 구현은 기존 공식 폴더 경로를 유지하면서 컴포넌트별 경로를 선택적으로 덮어쓰게 하는 것이다. 새 다운로드·자동 가중치 교체 없이 로컬 파일부터 지원하면 사용자가 이미 가진 ComfyUI 모델을 재사용하기 쉽다. Base와 SFT도 선택한 실제 체크포인트를 기준으로 기록하면 된다.

주의할 변환은 다음과 같다.

- ai-toolkit/ComfyUI DiT는 `proj.qkv`가 합쳐진 구조이고, 현재 모델은 `proj.q`, `proj.k`, `proj.v`가 분리되어 있다. 베이스 가중치를 현재 구조로 분리해서 읽으면 기존 252/269개 어댑터 범위를 유지할 수 있다.
- VAE는 이름뿐 아니라 FLUX.2 packing, BatchNorm 통계, decoder 구성까지 확인해야 한다. ai-toolkit은 별도 VAE와 서로 다른 decoder 구성을 처리하므로 기본 파일 선택을 그대로 복사하지 않는다.
- 현재 텍스트 캐시 식별자의 encoder/tokenizer 경로는 실제로 선택한 컴포넌트 경로를 따라야 한다.

이는 사용성과 호환성 개선이다. **단일 파일 입력 자체가 peak RAM을 낮춘다는 주장은 하지 않는다.** ai-toolkit도 로컬 Transformer 폴더 분기에서는 shard의 tensor를 하나의 state dict에 수집한다. [가중치 형식](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/extensions_built_in/diffusion_models/moss_image/src/transformer.py), [컴포넌트 선택 정책](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/extensions_built_in/diffusion_models/moss_image/src/checkpoints.py).

## 3. 양자화: 효과가 클 수 있지만 별도 구현·검증 필요

ai-toolkit의 MOSS 등록 기본값에는 DiT와 TE의 `convrot8` 양자화가 들어 있다. 실제 동작은 MOSS 파일 하나가 아니라 공통 loader mixin, quantizer, offload manager에 걸쳐 있다. 현재 sd-scripts MOSS trainer는 `fp8_base`와 `fp8_base_unet`을 명시적으로 거부한다. 이 검사만 해제해서 같은 기능을 얻을 수는 없다. [등록 기본값](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/toolkit/models/registry.py#L59-L62), [공통 정책](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/toolkit/models/v2/_mixin.py#L278-L452), [현재 제한](/home/munsy0227/sd-scripts/moss_train_network.py:49).

안전한 도입 순서는 아래와 같다.

1. **TE 양자화와 임베딩 캐시**: TE는 frozen이므로 어댑터 backward 경계와 분리해 평가하기 쉽다. 다만 임베딩 값은 달라질 수 있으므로 정밀도·backend·모델 revision·loader 방식을 캐시 식별자에 포함한다. 현재 [cache_identity](/home/munsy0227/sd-scripts/library/strategy_moss.py:39)에는 양자화 정책이 없다.
2. **DiT 선택적 양자화**: 기존 block swapping·checkpointing과 결합해서 실제 peak RAM/VRAM·속도를 비교한다. ai-toolkit처럼 text refiner, embedding, modulation, output head를 정밀도 유지 후보로 삼을 수 있다. 이는 후보 정책이며 최적 정책으로 검증된 것은 아니다. [양자화 제외 범위](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/extensions_built_in/diffusion_models/moss_image/src/transformer.py#L215-L224).
3. **LyCORIS 경계 확인**: 선택한 quantized Linear의 발견, base weight 복원, LoKR+DoRA norm 계산, adapter gradient, 저장·재개를 검증한다. 현재 [양자화 복원 코드](/home/munsy0227/LyCORIS/lycoris/utils/quant.py:76)가 있다는 사실만으로 ai-toolkit의 모든 backend가 호환된다고 볼 수 없다.

사용자의 full-matrix LoKR+DoRA에서는 FP32 magnitude와 norm 계산, full-matrix scale=1, portable checkpoint와 정확한 재개를 유지해야 한다. 양자화 base에 adapter를 영구 병합하는 기능은 명시적인 requantization 설계 없이는 확장하지 않는 편이 맞다.

이번에는 ai-toolkit 전체 실행, 양자화 모델 backward, 실제 생성 품질이나 VRAM 측정을 수행하지 않았다. 특정 GPU에서 학습이 가능해진다거나 품질을 유지한다는 수치를 제시할 근거는 없다.

## 4. QKV와 어댑터 호환성: LyCORIS core 교체보다 검증 우선

현재 LyCORIS는 `moss` 252개, `moss-full` 269개의 분리된 원래 경로를 대상으로 한다. 이 범위를 유지하는 편이 기존 학습과 ComfyUI 연결에 유리하다. [프리셋](/home/munsy0227/LyCORIS/lycoris/config.py:58).

ai-toolkit은 학습 모델의 QKV가 fused이며, LoRA 저장 시 output factor를 Q/K/V별로 나누고 input factor를 복제한다. 반대로 독립 Q/K/V LoRA를 읽을 때는 필요한 경우 rank를 세 배로 늘린 block 구조를 만든다. **이 변환은 LoKR Kronecker factor를 같은 방식으로 변환하는 기능이 아니다.** 실제 저장 코드는 LoKR 등 나머지 키를 fused 상태로 유지한다. [변환 코드](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/extensions_built_in/diffusion_models/moss_image/moss_image.py#L630-L683).

따라서 ai-toolkit LoKR+DoRA 파일과 현재 LyCORIS 파일이 prefix만 바꾸면 서로 재학습·재개 가능하다고 보아서는 안 된다. 우선 공통 정밀도의 동일 base에서 adapter 적용 출력을 비교하고, LoRA와 LoKR를 별도 형식으로 검증해야 한다. 특히 fused QKV를 도입해 LoKR factor의 행렬 모양 자체가 달라지면 학습 파라미터화도 달라진다.

권장 방향은 **베이스 파일 로더에서 QKV를 정규화하고 현재 학습 구조는 유지**하는 것이다. ai-toolkit과의 adapter 교환은 별도 호환 기능으로 다룬다.

## 이미 같은 방향인 부분과 보류할 부분

- **텍스트 인코딩**: 현재 코드와 ai-toolkit, 공식 pipeline 모두 user chat template, generation prompt, thinking 활성화, padding 없는 개별 forward, 마지막 hidden state를 사용한다. 이 경로는 교체할 이유를 찾지 못했다. [현재 인코딩](/home/munsy0227/sd-scripts/library/strategy_moss.py:52), [공식 pipeline](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/1ceede80badddd6a7812069619edaa5633587192/pipeline_moss_image.py#L314-L336).
- **학습 target와 latent**: 양쪽 모두 flow matching의 `noise - clean` 방향이며, FLUX.2의 2×2 packing과 BatchNorm 통계를 사용한다. 현재 flow 기본 grid와 ai-toolkit의 linear training grid가 같은 방향이다. 다만 ai-toolkit의 선택적 timestep 분포·loss weighting까지 동일하다고 주장하지 않는다.
- **샘플 sigma**: 현재 nearest-training-grid snapping은 ai-toolkit과 실제 tensor 비교에서 일치했다. ai-toolkit scheduler 생성자의 shift 값만 보고 현재 training shift를 5로 바꿀 근거는 없다. ai-toolkit의 `set_train_timesteps(linear/weighted)`는 별도로 linear grid를 설정한다. [샘플 grid](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/extensions_built_in/diffusion_models/moss_image/src/pipeline.py#L23-L48), [학습 grid](https://github.com/ostris/ai-toolkit/blob/9a1828f20131f3dc4fac8bab75bcfdf53df96398/toolkit/samplers/custom_flowmatch_sampler.py#L108-L120).
- **Qwen frozen 정책**: 현재 구현이 공식 권고와 일치한다. `moss-full`의 Transformer 내부 text adapter와 외부 Qwen text encoder는 구분해야 한다. [공식 학습 권고](https://github.com/OpenMOSS/MOSS-Image/blob/c1355e6b1986dfe8fba3c0455783a05899212342/USERGUIDE.md#lora-training-recommendations).
- **edit/reference 기능**: 이미지 conditioning, VLM 입력, reference latent, attention mask, 위치 처리, inference 연결까지 필요한 별도 기능이다. 지금 TXT/JSON 학습 경로 개선과 묶어서 가져올 우선순위는 낮다.

## 이번 검증과 제한

실제로 실행한 검사:

- LyCORIS `python -m unittest -q test.test_moss`: **7개 중 4개 통과, 3개 skip**. scope 선택, 유사 경로 제외, factory 상태 분리, prefix 저장·복원을 검사했다. 실제 모델 관련 skip을 통과로 계산하지 않았다.
- 현재 `MOSSImageScheduler`와 ai-toolkit의 `moss_sigmas` 함수를 독립적으로 실행해 steps 1/2/30/50 × shift 1/5의 **8개 sigma tensor가 정확히 일치**함을 확인했다. 기본 training sigma grid도 uniform grid와 비교했다.
- ai-toolkit의 QKV 변환 함수만 분리 실행해 shared-down LoRA와 independent Q/K/V LoRA의 유효 update가 보존됨을 확인했다. 후자의 fused rank는 4에서 12로 증가했다.
- 같은 변환의 저장 경로에서 fused LoKR/DoRA 예제 키가 fused 상태로 남는 것을 확인했다. 전체 ai-toolkit 설치나 trainer 실행 검증은 아니다.

런타임 제약:

- 로컬 sd-scripts `.venv`는 torch `2.13.0+cu130`, transformers `4.54.1`, diffusers `0.32.1`, safetensors `0.4.5`였으며 pytest가 없었다.
- 현재 저장소 요구 버전은 transformers `5.17.0`, diffusers `0.40.0`이다. 해당 `.venv`에서는 `Qwen3_5Config` import가 실패했다. 텍스트 전용 모델의 meta parameter count 측정도 이 때문에 완료하지 못했으며, 메모리 절감량을 숫자로 제시하지 않았다.
- sd-scripts MOSS pytest 실행 시도는 pytest 부재로 테스트 수집 전에 종료되었다. 전체 모델 학습·생성·CUDA 성능·양자화·Windows/MPS는 이번에 검증하지 않았다.
- 이전 기록의 모델 경로 `/run/media/munsy0227/MSY EXT4/MOSS-Image-Base`는 현재 존재하지 않았다. 공식 소스·config·checkpoint index를 온라인으로 확인했고, 가중치 본체는 내려받지 않았다.

실제 구현을 시작한다면 먼저 요구 버전에 맞는 별도 검증 환경을 준비하고, **텍스트 전용 로더 + 컴포넌트 경로 분리**를 첫 변경 단위로 삼는 것을 권한다. 그 단계에서 임베딩 일치·기존 cache 동작·기존 프리셋과 저장/재개를 확인한 후 양자화를 진행하면 된다.
