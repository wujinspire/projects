# Open-Source LLM Architecture Notes

## Summary Table

Representative-model comparison only.

| Model / representative | Type                            | Params (total / active)                             | Layers                    | Experts                                | Attention                                   | Norm                                       | Context        | Key features                                                                  | Detail score |
| ---------------------- | ------------------------------- | --------------------------------------------------- | ------------------------- | -------------------------------------- | ------------------------------------------- | ------------------------------------------ | -------------- | ----------------------------------------------------------------------------- | ------------ |
| DeepSeek V3            | MoE                             | 671B / 37B                                          | 61 = 3 dense + 58 MoE     | 1 shared + 256 routed; 8 active        | MLA                                         | RMSNorm                                    | 128K           | FP8 training, auxiliary-loss-free load balancing, MTP depth = 1               | 100%         |
| DeepSeek V3.2          | Delta report over V3.1-Terminus | Not fully restated in the delta report              | Not explicitly specified  | Not explicitly specified               | DSA under MLA, MQA-mode kernel              | Not explicitly specified                   | 128K           | Continued pre-training, dense warm-up + sparse adaptation, Keep Routing in RL | 60%          |
| Qwen3-235B-A22B        | MoE                             | 235B / 22B                                          | 94                        | 128 total; 8 active; no shared experts | GQA                                         | RMSNorm, pre-norm                          | 128K           | QK-Norm, no QKV bias, ABF + YARN + DCA                                        | 95%          |
| Qwen3.5-397B-A17B      | Multimodal MoE                  | 397B / 17B                                          | 60                        | 512 total; 10 routed + 1 shared        | Gated DeltaNet + Gated Attention            | Not explicitly specified                   | 262,144 / 1.01M | Hybrid linear-attention stack, multimodal, multi-step MTP, default thinking   | 85%          |
| Gemma 3 27B            | Dense; mostly multimodal family | 27B / -                                             | Not explicitly specified  | -                                      | GQA with 5:1 local/global attention         | RMSNorm; pre- and post-norm both mentioned | 128K           | QK-norm, SigLIP vision encoder, local/global interleaving                     | 75%          |
| gpt-oss-120b           | MoE                             | 116.83B / 5.13B                                     | 36                        | 128 total; top-4 active                | GQA with alternating banded/dense attention | RMSNorm, Pre-LN                            | 131,072        | MXFP4 MoE quantization, learned attention sink bias                           | 80%          |
| Llama 3.1 405B         | Dense                           | 405B / -                                            | 126                       | -                                      | GQA                                         | Not explicitly specified                   | 128K           | RoPE theta = 500,000, 128K vocab                                              | 70%          |
| Ministral 3 14B        | Dense multimodal                | 14B / -                                             | 40                        | -                                      | GQA                                         | RMSNorm                                    | up to 256K     | YaRN, position-based softmax temperature scaling, frozen ViT encoder          | 75%          |
| GLM-4.5                | MoE                             | 355B / 32B                                          | 3 dense + 89 MoE (+1 MTP) | 160 total; 8 active; 1 shared          | GQA with partial RoPE                       | Not explicitly specified                   | 131,072        | QK-Norm, loss-free balance routing, MTP                                       | 90%          |
| GLM-5                  | MoE                             | 744B / 40B                                          | 3 dense + 75 MoE (+1 MTP) | 256 total; 8 routed; 1 shared          | MLA + DSA                                   | Not explicitly specified                   | 200K / 202,752 | Indexer heads, Q/KV LoRA dims, long-context focus                             | 90%          |
| Kimi K2                | MoE                             | 1.04T / 32.6B                                       | 61                        | 384 total; 8 active; 1 shared          | MLA                                         | Not explicitly specified                   | 128K           | MuonClip, QK-Clip, only 1 dense layer                                         | 90%          |
| Kimi K2.5              | Multimodal system               | Full system not fully restated; K2 backbone = 1.04T / 32B | Not explicitly specified  | Not explicitly specified               | MoonViT-3D + K2 backbone                    | Not explicitly specified                   | 256K / 262,144 | Visual agentic stack, NaViT-style packing, Agent Swarm / PARL                 | 65%          |

## Cross-Model Takeaways

- Attention choice: `GQA` is the default in many dense or moderately sparse open models here (`Qwen 3`, `Gemma 3`, `Llama 3.1`, `Ministral 3`, `GLM-4.5`), while `MLA` is favored by several frontier MoE designs (`DeepSeek V3`, `GLM-5`, `Kimi K2`) to reduce KV-cache cost.
- Attention evolution: long-context improvements are often added on top of a base attention choice rather than replacing the whole stack, e.g. `DSA` over `MLA`, local/global interleaving over `GQA`, banded attention over dense attention, and inference extensions such as `YaRN`, `ABF`, or `DCA`.
- Hybrid frontier trend: `Qwen 3.5` shows a different path from plain `GQA` or `MLA`, combining `Gated DeltaNet` with sparse MoE and a smaller gated-attention component in a single multimodal stack.
- Norm choice: `RMSNorm` is the dominant normalization choice across almost all families in this set; when reports are explicit, modern stacks usually use pre-norm-like layouts or add extra normalization around attention internals.
- Attention stabilization: several families add explicit Q/K-side stabilization instead of changing the full block design, e.g. `QK-Norm` in `Qwen 3`, `Gemma 3`, and `GLM-4.5`, plus `QK-Clip` in `Kimi K2`.
- MoE choice: the largest open frontier models in this set mostly adopt `MoE` to push total parameter count up while keeping activated parameters per token much lower than total parameters.
- Common MoE pattern: a small number of dense layers is often kept near the bottom of the stack, then most later layers use routed experts; many reports converge around `top-8` routed or active experts, sometimes with `1 shared expert`.
- Dense-model tradeoff: dense families (`Llama 3.1`, much of `Gemma 3`, `Ministral 3`) expose fewer routing knobs and are easier to summarize architecturally, but they usually scale active compute linearly with model size instead of using sparse activation.
- Multimodal trend: multimodal systems in this collection usually keep the language backbone mostly intact and add a frozen or separately trained vision encoder plus a projector, instead of redesigning the core Transformer from scratch.

## Tunable Parameters

- In this note, `tunable parameters` means architecture and system knobs that model designers can choose or sweep, not the full set of trainable weights.
- Backbone scale knobs: number of layers, hidden dimension, FFN/intermediate dimension, vocabulary size, and whether embeddings are tied.
- Attention knobs: attention type (`GQA`, `MLA`, local/global, banded, `DSA`), query-head count, KV-head count, per-head dimension, compression dimensions, and any indexer or latent-attention sub-dimensions.
- Positional / context knobs: context length, RoPE base frequency or theta, local attention window size, local/global layer ratio, and long-context extension method such as `YaRN`, `ABF`, or `DCA`.
- MoE knobs: number of experts, number of routed experts per token, number of shared experts, expert hidden dimension, number of dense vs MoE layers, grouping / routing constraints, and load-balancing strategy.
- Routing knobs: top-k value, sigmoid vs other gate details, node limits, auxiliary-loss vs auxiliary-loss-free balancing, and whether routing paths are kept fixed in special training stages.
- Extra-module knobs: `MTP` depth, LoRA-style compression dimensions, indexer heads for sparse attention, projector size for multimodal models, and vision-token compression settings.
- System-level knobs: which backbone is frozen or reused, whether the vision encoder is frozen, sequence packing strategy, temporal pooling / compression, and how agent or tool-use components are layered on top of the base LLM.

## Details

This document records architecture facts conservatively from the technical reports already collected in `os_llms/`.

Working rules:

- Write model notes first, then move verified facts into the comparison table.
- If a paper does not explicitly report a field, leave it as `Not explicitly specified`.
- `Dense layers` means Transformer layers that use dense FFNs instead of expert routing.
- For dense-only models, write `Expert layers = 0`.
- For family papers, the final comparison table uses one representative model, usually the flagship model or the largest released model; the section body keeps the family context.

## DeepSeek V3

### Core Architecture

- Model type: Mixture-of-Experts (MoE)
- Total parameters: 671B
- Activated parameters: 37B per token
- Transformer layers: 61
- Dense layers: 3
- Expert layers: 58
- MoE design: DeepSeekMoE
- Experts per MoE layer: 1 shared expert + 256 routed experts
- Activated experts: 8 routed experts per token
- Node-limited routing: each token is sent to at most 4 nodes

### Attention, Norm, and Dimensions

- Attention type: Multi-head Latent Attention (MLA)
- Attention heads: 128
- Per-head dimension: 128
- KV compression dimension: 512
- Query compression dimension: 1536
- Decoupled Q/K RoPE head dimension: 64
- Positional encoding: RoPE
- Normalization: RMSNorm
- Normalization setup: RMSNorm is explicitly used, with additional RMSNorm layers after the compressed latent vectors; the report does not explicitly label the block layout as pre-norm or post-norm in text
- Residual setup: standard residual addition is present in the block computations; the report does not explicitly describe any special residual variant or residual scaling setup
- Extra normalization detail: additional RMSNorm layers after compressed latent vectors
- Hidden dimension: 7168

### FFN, Tokenizer, and Training Setup

- Expert intermediate dimension: 2048
- Tokenizer: byte-level BPE
- Vocabulary size: 128K
- Pre-training tokens: 14.8T
- Pre-training sequence length: 4K
- Long-context extension: 32K, then 128K
- Precision / training stack: FP8 mixed precision training
- Routing detail: sigmoid gating with top-K affinity normalization
- Extra training objective: Multi-Token Prediction (MTP), depth = 1
- Load balancing: auxiliary-loss-free load balancing

## Qwen 3

- Paper scope: family of 6 dense models and 2 MoE models
- Representative model for comparison: Qwen3-235B-A22B
- Model type: MoE
- Total parameters: 235B
- Activated parameters: 22B per token
- Transformer layers: 94
- Dense layers: Not explicitly specified
- Expert layers: Not explicitly specified
- Experts per MoE model: 128 total experts, 8 activated experts per token
- Shared experts: none
- Attention type: Grouped Query Attention (GQA)
- Attention heads: 64 query heads
- Key-value heads: 4
- Head dimension: Not explicitly specified
- Positional encoding: RoPE
- RoPE / long-context setup: base frequency increased from 10,000 to 1,000,000 with ABF; YARN and Dual Chunk Attention (DCA) are used for longer-context inference
- Normalization: RMSNorm
- Normalization setup: pre-normalization
- Residual setup: Not explicitly specified
- FFN activation: SwiGLU
- Attention-specific setup: QK-Norm is added; QKV bias is removed
- Tokenizer: Qwen tokenizer with byte-level BPE (BBPE)
- Vocabulary size: 151,669
- Context length: 128K for the representative model
- Pre-training data scale: 36T tokens
- Routing / training notes: fine-grained expert segmentation; no shared experts; global-batch load balancing loss

## Qwen 3.5

- Source scope: model card / release page for `Qwen3.5-397B-A17B`; this is not a full technical paper
- Representative model for comparison: `Qwen3.5-397B-A17B`
- Model type: multimodal MoE language model with vision encoder
- Total parameters: 397B
- Activated parameters: 17B
- Transformer layers: 60
- Dense layers: Not explicitly specified
- Expert layers: Not explicitly specified
- Hidden layout: `15 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))`
- Hidden dimension: 4096
- Attention type: hybrid linear-attention-plus-attention stack with Gated DeltaNet and Gated Attention
- Linear-attention heads: 64 for V and 16 for QK
- Gated-attention heads: 32 query heads and 2 key-value heads
- Head dimension: 128 in Gated DeltaNet; 256 in Gated Attention
- Rotary position embedding dimension: 64
- Experts per MoE model: 512 total experts
- Activated experts: 10 routed + 1 shared
- Expert intermediate dimension: 1024
- Token embedding / LM output size: 248320 padded
- Context length: 262,144 natively; extensible up to 1,010,000 tokens
- Extra training objective: MTP trained with multi-steps
- Extra notes: model card states thinking mode is enabled by default; the hosted `Qwen3.5-Plus` variant is described separately with additional production features

## Gemma 3

- Paper scope: family of 1B, 4B, 12B, and 27B models
- Representative model for comparison: Gemma 3 27B
- Model type: dense model family; most models are multimodal, while the 1B variant is not presented with the same vision setup as the 4B/12B/27B models
- Total parameters: family ranges from 1B to 27B; representative model is 27B
- Activated parameters: Not explicitly specified
- Transformer layers: Not explicitly specified
- Dense layers: Not explicitly specified
- Expert layers: 0
- Attention type: GQA with local sliding-window self-attention and global self-attention
- Attention pattern: 5 local layers for every 1 global layer, starting with a local layer
- Local attention span: 1024 tokens
- Head counts and head dimension: Not explicitly specified
- Positional encoding: RoPE
- RoPE setup: global layers use base frequency 1M; local layers keep base frequency 10K
- Normalization: RMSNorm
- Normalization setup: the report explicitly mentions both post-norm and pre-norm with RMSNorm
- Residual setup: Not explicitly specified
- Attention-specific setup: QK-norm replaces Gemma 2 soft-capping
- Tokenizer: SentencePiece tokenizer, same tokenizer family as Gemini 2.0, with split digits, preserved whitespace, and byte-level encodings
- Vocabulary size: 256k in the Table 1 caption; 262k in the tokenizer prose
- Context length: 128K for Gemma 3 models except the 1B model, which has 32K
- Vision setup: 400M SigLIP vision encoder with 896 x 896 input resolution
- Extra notes: decoder-only family with local/global attention interleaving; multimodality is central for most models, but the 1B variant is not described with the same shared SigLIP setup as 4B/12B/27B

## gpt-oss

- Paper scope: family of two released models, `gpt-oss-120b` and `gpt-oss-20b`
- Representative model for comparison: `gpt-oss-120b`
- Model type: MoE
- Total parameters: 116.83B
- Activated parameters: 5.13B per token per forward pass
- Transformer layers: 36
- Dense layers: Not explicitly specified
- Expert layers: Not explicitly specified
- Experts per MoE block: 128
- Activated experts: top-4 experts per token
- Hidden dimension / residual stream dimension: 2880
- Attention type: GQA with alternating banded-window and fully dense attention
- Attention heads: 64 query heads
- Key-value heads: 8
- Head dimension: 64
- Attention-specific setup: bandwidth 128 for banded attention; each attention head has a learned bias in the softmax denominator
- Positional encoding: RoPE
- Long-context setup: YaRN extends dense-layer context to 131,072 tokens
- Normalization: RMSNorm
- Normalization setup: Pre-LN; RMSNorm is applied before each attention and MoE block
- Residual setup: residual stream is explicit; the paper also notes that the SwiGLU implementation includes an additional residual connection
- FFN / expert activation: gated SwiGLU
- Tokenizer: `o200k_harmony` BPE tokenizer
- Vocabulary size: 201,088
- Context length: 131,072
- Extra notes: MoE weights are post-trained to MXFP4 format

## Llama 3

- Paper scope: family of 8B, 70B, and 405B language models; benchmark reporting is for the Llama 3.1 models
- Representative model for comparison: Llama 3.1 405B
- Model type: dense Transformer
- Total parameters: 405B trainable parameters for the representative model
- Activated parameters: Not explicitly specified
- Transformer layers: 126
- Dense layers: 126
- Expert layers: 0
- Hidden dimension: 16,384
- Attention type: GQA
- Attention heads: 128
- Key-value heads: 8
- Head dimension: Not explicitly specified
- Positional encoding: RoPE
- RoPE setup: theta = 500,000
- Normalization: Not explicitly specified
- Normalization setup: Not explicitly specified
- Residual setup: Not explicitly specified
- FFN activation: SwiGLU
- FFN dimension: 53,248
- Tokenizer: tiktoken-based vocabulary with 100K inherited tokens plus 28K additional tokens
- Vocabulary size: 128,000
- Context length: up to 128K
- Extra notes: dense-only design; the paper explicitly contrasts this choice with MoE

## Ministral 3

- Paper scope: family of 3B, 8B, and 14B models
- Representative model for comparison: Ministral 3 14B
- Model type: dense decoder-only Transformer with a frozen vision encoder for image understanding
- Total parameters: 14B for the representative model
- Activated parameters: Not explicitly specified
- Transformer layers: 40
- Dense layers: 40
- Expert layers: 0
- Hidden dimension: 5120
- Attention type: GQA
- Attention heads: 32 query heads
- Key-value heads: 8
- Head dimension: Not explicitly specified
- Positional encoding: RoPE
- Long-context setup: YaRN and position-based softmax temperature scaling
- Normalization: RMSNorm
- Normalization setup: Not explicitly specified
- Residual setup: Not explicitly specified
- FFN activation: SwiGLU
- FFN dimension: 16384
- Tokenizer: Not explicitly specified
- Vocabulary size: 131K
- Context length: up to 256K; reasoning variants are reported as 128K
- Embedding setup: the 3B model uses tied input-output embeddings; the representative 14B model does not

## DeepSeek V3.2

- Paper scope: family of DeepSeek-V3.2 variants, with the main report centered on DeepSeek-V3.2 and its continued training from DeepSeek-V3.1-Terminus
- Representative model for comparison: DeepSeek-V3.2
- Model type: backbone details are mostly inherited; the report itself does not restate the full numeric backbone specification
- Total parameters: Not explicitly specified in this report
- Activated parameters: Not explicitly specified in this report
- Transformer layers: Not explicitly specified in this report
- Dense layers: Not explicitly specified in this report
- Expert layers: Not explicitly specified in this report
- Attention type: DeepSeek Sparse Attention (DSA) instantiated under MLA
- Attention kernel mode: DSA is implemented under the MQA mode of MLA
- Positional encoding: RoPE is shown in the architecture figures
- Normalization: Not explicitly specified
- Normalization setup: Not explicitly specified
- Residual setup: Not explicitly specified
- Tokenizer: Not explicitly specified
- Vocabulary size: Not explicitly specified
- Context length: 128K
- Routing / training notes: continued pre-training from DeepSeek-V3.1-Terminus; DSA uses a dense warm-up stage and a sparse adaptation stage; the report also introduces Keep Routing for MoE RL stability
- Extra notes: this report is best read as a delta report over DeepSeek-V3.1-Terminus rather than a full restatement of the whole backbone

## GLM-4.5

- Paper scope: family of GLM-4.5 and GLM-4.5-Air
- Representative model for comparison: GLM-4.5
- Model type: MoE
- Total parameters: 355B
- Activated parameters: 32B
- Layer split: 3 dense layers, 89 MoE layers, and 1 MTP layer
- Hidden dimension: 5120
- Dense intermediate dimension: 12288
- MoE intermediate dimension: 1536
- Attention type: GQA with partial RoPE
- Attention heads: 96
- Key-value heads: 8
- Attention head dimension: 128
- Positional encoding: partial RoPE
- RoPE setup: base frequency is adjusted from 10,000 to 1,000,000 during long-context extension
- Normalization: RMSNorm is mentioned in optimizer exclusions, but the block-level normalization setup is not explicitly specified
- Normalization setup: Not explicitly specified
- Residual setup: Not explicitly specified
- Experts per model: 160 total experts
- Activated experts: 8 experts per token
- Shared experts: 1
- Attention-specific setup: QK-Norm is enabled
- Context length: sequence length extends from 4,096 to 32,768 and then to 131,072 during training
- Tokenizer: Not explicitly specified
- Vocabulary size: Not explicitly specified in `glm4_5.md`
- Routing / training notes: loss-free balance routing, sigmoid gates, and 1 MTP layer

## GLM-5

- Paper scope: single flagship model
- Representative model for comparison: GLM-5
- Model type: MoE
- Total parameters: 744B
- Activated parameters: 40B
- Layer split: 3 dense layers, 75 MoE layers, and 1 MTP layer
- Hidden dimension: 6144
- Dense intermediate dimension: 12288
- MoE intermediate dimension: 2048
- Attention type: MLA with DSA
- Attention heads: 64
- Key-value heads: not reported numerically in the architecture table
- QK head dimension: 192
- V head dimension: 256
- Indexer attention heads: 32
- Indexer head dimension: 128
- Positional encoding: Not explicitly specified
- Normalization: Not explicitly specified
- Normalization setup: Not explicitly specified
- Residual setup: Not explicitly specified
- Experts per model: 256 total experts
- Routed experts: 8
- Shared experts: 1
- Vocabulary size: 154880
- Context length: the model is extended to 200K / 202,752-token settings in the report
- Attention-specific setup: Q LoRA dim = 2048; KV LoRA dim = 512
- Extra notes: DSA is a central architecture change for long-context efficiency

## Kimi K2

- Paper scope: single flagship model
- Representative model for comparison: Kimi K2
- Model type: MoE
- Total parameters: 1.04T in the architecture table; the abstract also refers to 1 trillion total parameters
- Activated parameters: 32.6B in the architecture table; the abstract also rounds this to 32B
- Transformer layers: 61
- Dense layers: 1
- Expert layers: 60, inferred from 61 total layers and 1 dense layer in the architecture table
- Hidden dimension: 7168
- MoE expert hidden dimension: 2048
- Attention type: MLA
- Attention heads: 64
- Key-value heads: Not explicitly specified
- Head dimension: Not explicitly specified
- Positional encoding: rotary components are discussed in the QK-Clip section, but a single RoPE base frequency is not reported
- Normalization: Not explicitly specified
- Normalization setup: Not explicitly specified
- Residual setup: Not explicitly specified
- Experts per model: 384 total experts
- Activated experts: 8 experts per token
- Shared experts: 1
- FFN activation: SwiGLU
- Tokenizer: Not explicitly specified
- Vocabulary size: Not explicitly specified
- Context length: 4,096-token pre-training context, long-context stages around 32K, and 128K extension with YaRN
- Training / stability notes: MuonClip optimizer and QK-Clip are central design points

## Kimi K2.5

- Paper scope: single multimodal system built on top of the Kimi K2 language backbone
- Representative model for comparison: Kimi K2.5
- Model type: multimodal system with MoonViT-3D vision encoder, MLP projector, and Kimi K2 language backbone
- Total parameters: not explicitly specified for the full end-to-end system; the cited Kimi K2 backbone is 1.04T total
- Activated parameters: not explicitly specified for the full end-to-end system; the cited Kimi K2 backbone is 32B activated
- Transformer layers: Not explicitly specified in `kimi_k2_5.md`
- Dense layers: Not explicitly specified in `kimi_k2_5.md`
- Expert layers: Not explicitly specified in `kimi_k2_5.md`
- Attention / visual setup: NaViT-style packing in MoonViT-3D; 4 consecutive frames are packed as a spatiotemporal volume
- Temporal compression: MoonViT-3D applies temporal pooling for 4x temporal compression before projection
- Language backbone: Kimi K2 is the underlying MoE LLM
- Normalization: Not explicitly specified
- Normalization setup: Not explicitly specified
- Residual setup: Not explicitly specified
- Tokenizer: Not explicitly specified
- Vocabulary size: Not explicitly specified
- Context length: sequence lengths of 32,768 to 262,144 appear in the training setup; 256K is the practical headline context
- Training / system notes: large-scale visual-text joint pre-training; Agent Swarm / PARL style orchestration for agentic behavior
- Extra notes: this report is more about the multimodal and agentic stack than a full reprint of the K2 backbone specification
