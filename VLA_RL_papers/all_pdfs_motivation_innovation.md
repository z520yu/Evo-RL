# pdfs 全部论文的 motivation 与核心创新梳理

范围：读取 `pdfs/` 下 34 个 PDF 后整理。这里把 model card、survey、benchmark paper 也纳入，因为它们能解释这批论文共同围绕的问题。

## 一、逐篇速读表

### A. 数据、基础 VLA 与工程效率

| PDF | 论文/模型 | Motivation | 核心创新点 |
|---|---|---|---|
| `open_x_embodiment_2310.08864.pdf` | Open X-Embodiment | 机器人数据长期分散在单机器人、单实验室、单场景里，难以训练可迁移的 generalist policy。 | 组织跨机构、跨 embodiment 的 Open X-Embodiment 数据集，并训练 RT-X 系列模型证明 cross-embodiment 数据能带来正迁移。 |
| `octo_2405.12213.pdf` | Octo | 通用机器人策略需要能适配不同传感器、动作空间和任务形式，且要开源可复现。 | 在 OXE 约 800k 轨迹上预训练 transformer policy，支持语言/目标图像、灵活 observation/action readout、diffusion action head，以及快速 finetuning 到新机器人接口。 |
| `openvla_2406.09246.pdf` | OpenVLA | 早期 VLA 多为闭源，且缺乏低成本适配新任务/新机器人配方。 | 7B 开源 VLA，基于 Prismatic/Llama2、SigLIP+DINOv2 视觉特征、970k 机器人轨迹；系统研究 LoRA、量化和高效 finetuning。 |
| `pi0_2410.24164.pdf` | pi0 | 机器人 foundation model 需要同时具备语义泛化和高频连续控制能力，离散动作 token 不够适合灵巧任务。 | VLM backbone + flow-matching action expert + action chunking；用大规模跨机器人数据预训练，再 post-train 到洗衣、清桌、装盒等复杂任务。 |
| `pi05_2504.16054.pdf` | pi0.5 | 真正有用的机器人要离开实验室，在未见过的家庭中完成长程任务，单一机器人示教覆盖不了开放世界。 | 用 heterogeneous co-training 融合移动机器人数据、其他机器人数据、web 多模态任务、高层 subtask prediction、语言指导；形成高层 subtask + 低层 action 的层级 VLA。 |
| `pi06_model_card.pdf` | pi0.6 Model Card | 希望在不做任务特定 finetuning 的情况下，比 pi0.5 有更强 out-of-box 表现。 | Gemma3 4B + SigLIP backbone，约 860M action expert，最多 4 路图像，Knowledge Insulation，metadata conditioning，更丰富数据，静态/移动/泛化任务均提升。 |
| `lingbot_vla_2601.18692.pdf` | LingBot-VLA | 领域缺少真实机器人数据 scaling law 和高吞吐训练栈，难判断“多收真实数据”是否持续有效。 | 约 20,000 小时真实双臂数据、9 种双臂配置；在 3 平台各 100 任务评测；同时贡献高吞吐开源训练代码。 |
| `smolvla_2506.01844.pdf` | SmolVLA | 主流 VLA 太大、训练和部署成本高，低成本机器人社区数据未被充分利用。 | 小型 VLM + flow-matching action expert；VLM layer skipping、少量视觉 token、交错 cross/self attention；少于 30k 社区 episode 训练，单 GPU 可训，CPU/消费级 GPU 可部署，并做异步推理。 |
| `streamingvla_2603.28565.pdf` | StreamingVLA | VLA 的 observation、action generation、execution 串行导致延迟高、执行停顿，尤其边缘设备上明显。 | 把 VLA 执行改成 streaming 异步流水线；action flow matching 让动作边生成边执行，adaptive early observation 用 action saliency 决定何时提前观测；报告 2.4x 延迟提升和 6.5x 停顿降低。 |
| `focusvla_2603.28740.pdf` | FocusVLA | 自回归 VLA 不是视觉表征不够好，而是 action 生成时没有有效利用任务相关视觉细节。 | 诊断 architectural shortcut、visual token overload、task-irrelevant noise 三个瓶颈；提出 Modality Cascaded Attention 和 Focus Attention，在 patch/channel 级聚焦任务相关视觉信息。 |

### B. 推理、世界模型与视觉反馈

| PDF | 论文/模型 | Motivation | 核心创新点 |
|---|---|---|---|
| `rynnvla002_2511.17502.pdf` | RynnVLA-002 | VLA 会出动作但缺少“想象/物理预测”，world model 会预测未来但不能直接规划动作。 | 统一 VLA 与 world model 为 action world model，共享 image/text/action/state token 空间；加入 action attention mask 缓解离散 action chunk 误差累积，并加连续 Action Transformer 提升真实机器人泛化和平滑性。 |
| `vla_r1_2510.01623.pdf` | VLA-R1 | 现有 VLA 多直接输出动作，缺少 affordance、几何关系、轨迹约束上的显式 step-by-step reasoning。 | 构造 VLA-CoT-13K 数据引擎；用 RLVR + GRPO 后训练，奖励包括区域对齐、轨迹一致性、输出格式，联合强化推理与执行。 |
| `thinkact_2507.16815.pdf` | ThinkAct | 端到端 VLA 对长程任务和复杂变化缺少显式规划，纯 CoT SFT 又依赖昂贵标注。 | 双系统框架：MLLM 通过 goal completion 和 trajectory consistency 的 action-aligned visual reward 学会 embodied reasoning；再把 reasoning 压缩为 visual plan latent 去条件化动作模型。 |
| `sole_r1_2603.28730.pdf` | SOLE-R1 | 通用 VLM 当 RL rewarder 时会受 partial observability 和 distribution shift 影响，被策略 reward hacking。 | 训练 video-language reasoning model，逐时刻输出 CoT 和 progress score，把 progress 直接作为 dense reward；用视频轨迹合成 + SFT + RLVR，支持无真实奖励/无示教的 zero-shot online RL。 |

### C. RL、后训练与从 generalist 到 specialist

| PDF | 论文/模型 | Motivation | 核心创新点 |
|---|---|---|---|
| `hil_serl_2410.21845.pdf` | HIL-SERL | 真实机器人 RL 有样本效率、稳定性、奖励和安全问题，但它有机会超过示教和手工控制。 | 把示教、人类 correction、RLPD/off-policy RL、预训练视觉 backbone、安全低层控制器整合成真实世界系统；数小时内学会双臂、动态、精密装配等任务。 |
| `in_ril_2505.10442.pdf` | IN-RIL | IL 稳定但泛化弱，RL 可探索但不稳定；传统 IL pretrain 后 RL finetune 容易 collapse。 | 在 RL 更新之间周期性插入 IL 更新；提出 gradient surgery 和 residual/network separation 防止 IL/RL 梯度冲突，并作为多种 RL 算法的 plug-in。 |
| `rpd_vla_to_rl_experts_2503.05833.pdf` | Refined Policy Distillation | VLA 泛化强但成功率/速度常不如专用 expert，RL 从零学又样本低效。 | 用 VLA teacher 的动作指导 PPO student，加入 BC/MSE teacher action loss；student 通过 RL 探索超过 teacher，得到小而快的 task-specific expert。 |
| `vlajs_2604.13733.pdf` | VLAJS | 高频 state-based RL 精确但长程稀疏奖励 credit assignment 差；VLA 语义强但低频、精度和延迟有限。 | 用 VLA 作为稀疏、低频、早期指导信号，在 PPO 中加 direction-only action consistency regularization，并随 reward improvement 退火，避免永久模仿限制上限。 |
| `rm_rl_2510.15189.pdf` | RM-RL | 高精度真实操作很难收集高质量示教，offline RL 有分布偏移，online RL 数据利用率低。 | 在相似初始状态下选最高 reward 的“role-model action”，自动给在线数据打监督标签；线上采样 + 离线监督 replay 混合，提高精密操作收敛速度和精度。 |
| `rl100_2510.14830.pdf` | RL-100 | 纯模仿有 imitation ceiling，部署需要接近或超过人类的可靠性、效率、鲁棒性。 | 扩散视觉运动策略三阶段训练：IL 预训练、迭代 offline RL、少量 targeted online RL；统一 clipped PPO surrogate 作用在 denoising 过程，并用 consistency distillation 压到一步控制。 |
| `rlt_pi_rl_token.pdf` | RL Token / RLT | 大 VLA 通用但在最后毫米级精度和速度上常失败，端到端 RL 微调整个 VLA 太贵。 | 让 VLA 暴露一个紧凑 RL token；冻结 VLA，用小 actor-critic 基于 RL token 和 VLA reference action chunk 做在线 RL，并用 anchor 保持接近 VLA。 |
| `pistar06_recap_2511.14759.pdf` | pi*0.6 / RECAP | VLA 要从部署经验中持续进步，不能只依赖离线示教；失败、correction、自主经验都应被利用。 | 用 value/advantage 训练 advantage-conditioned VLA，把 demonstrations、human corrections、autonomous rollouts 统一进 offline RL；可多轮迭代提升真实任务吞吐和失败率。 |
| `gr_rl_2512.01801.pdf` | GR-RL | 长程高精度灵巧任务中，人类示教含犹豫、错误和亚优片段，直接模仿会限制 specialist。 | 多阶段流程：offline RL 学 Q/progress 并过滤示教、形态对称增强、再用 online RL 学 latent/noise predictor 对齐部署行为；展示系鞋带等长程任务。 |
| `vla_rl_2505.18719.pdf` | VLA-RL | VLA 只利用离线示教会在 OOD 状态失败，需要 scalable online RL 打开 test-time exploration。 | 把自回归 VLA 轨迹建模成 multimodal multi-turn conversation；训练 robotic process reward model 缓解稀疏奖励；配合 curriculum、GPU-balanced vectorized env、batch decoding、critic warmup。 |
| `simplevla_rl_2509.09674.pdf` | SimpleVLA-RL | SFT 受数据稀缺和分布转移限制，受 LRM/R1 启发，想看 RL 能否提升 VLA step-by-step action planning。 | 基于 veRL 做 VLA-specific online RL：interactive trajectory sampling、并行渲染、多环境、优化 loss；报告少示教、大幅泛化提升和 RL 发现 “pushcut” 新行为。 |
| `co_rft_2508.02219.pdf` | CO-RFT | VLA RL finetuning 面临样本效率、训练稳定性和 action chunking 兼容性问题。 | 提出 Chunked RL：critic 对 action chunk 预测一串 Q 值并做 TD；CO-RFT 先全参 IL 适配 workspace/embodiment，再用 30-60 条 demo 做 offline chunked RL。 |
| `vla_rft_2510.00406.pdf` | VLA-RFT | IL 有 compounding error，真实 RL 贵且危险，仿真 RL 有 sim-to-real gap，offline RL 又不能从自己动作后果中学习。 | 用数据驱动 world model 作为可控 simulator，rollout VLA action sequence，依据目标参考轨迹构造 pixel/perceptual verified reward，并用 GRPO 微调 flow/action policy。 |
| `world_env_2509.24948.pdf` | World-Env | 真实环境不可轻易 reset，交互昂贵/危险；VLA 也缺少可靠 done/termination 判断。 | world model virtual environment + geometry-aware feature injection；VLM instant reflector 给 dense reward 和 done signal；结合 demo 和 VLA 自探索训练。 |
| `vlac_2509.15937.pdf` | VLAC | 真实 VLA RL 被稀疏手写奖励和低效探索卡住。 | 基于 InternVL 训练通用 process reward model，输入相邻/成对观测和语言目标，输出 progress delta 与 done；同一 autoregressive 模型可按 prompt 充当 critic 或 actor，并配合异步真实 RL 和分级 HIL。 |
| `irl_vla_2508.06571.pdf` | IRL-VLA | 自动驾驶 VLA 的 open-loop IL 只能复现数据，closed-loop RL 又依赖昂贵高保真仿真且有 domain gap。 | 三阶段：VLA imitation pretraining、inverse RL reward world model、PPO closed-loop fine-tuning；用 learned RWM 替代重仿真奖励计算。 |

### D. 评测、综述与基准化

| PDF | 论文/模型 | Motivation | 核心创新点 |
|---|---|---|---|
| `vla_eval_2603.13966.pdf` | vla-eval | VLA 评测受 benchmark 依赖冲突、协议不清、预处理差异影响，N 个模型 x M 个 benchmark 成本很高。 | 用 Docker 隔离 benchmark、WebSocket+msgpack 解耦模型服务与环境；模型只实现一次 `predict()`，支持 14 个 benchmark、6 个模型 server、episode sharding/batch inference 和 leaderboard。 |
| `maniparena_2603.28545.pdf` | ManipArena | 仿真 benchmark 难反映真实接触、感知噪声、硬件延迟；真实评测又分散在不同平台，难公平比较。 | 标准化真实评测框架：20 个 reasoning-oriented 任务、10,812 expert trajectories、多级 OOD、移动操作、低层电机/传感诊断、Real2Sim，同一模型跑所有任务。 |
| `robotarena_inf_2510.23571.pdf` | RobotArena infinity | 真实机器人评测慢、贵、不安全且不可复现；通用策略需要可持续扩展的排名体系。 | 自动把真实视频 demo 转成仿真环境，利用 VLM/2D-to-3D/differentiable rendering；用 VLM 分数和人类 pairwise preference 评价，系统扰动背景/物体位置等，形成大规模可演化 benchmark。 |
| `vla_survey_2507.10672.pdf` | VLA Systematic Review | VLA 模型、数据集、仿真平台增长很快，需要系统 taxonomy 来定位趋势和空白。 | 系统梳理 102 个 VLA 模型、26 个数据集、12 个仿真平台；提出架构分类和数据集二维量化框架，按 task complexity 与 multimodal richness 找数据空白。 |

## 二、这些论文的共同规律

### 1. 核心矛盾从“有没有 VLA”变成“VLA 怎么后训练”

早期主线是 Open X-Embodiment、Octo、OpenVLA、pi0：证明大规模多机器人数据 + VLM/VLA backbone 能做 generalist policy。后面几乎所有论文都在补同一个短板：SFT/BC 只能复现示教分布，遇到 OOD、长程任务、精密接触和失败恢复就不够。

所以新论文的标准开场基本一致：

- demos 昂贵且覆盖窄；
- imitation 有 compounding error 和 imitation ceiling；
- VLA 泛化强但不够可靠；
- RL 能从经验中提升，但机器人 RL 太贵、太稀疏、太不稳定。

### 2. RL 不是直接套 PPO，而是围绕三个瓶颈做“保护层”

VLA + RL 的难点通常不在“选择 PPO/GRPO/SAC”，而在三件事：

- 奖励：稀疏终止奖励无法训练长程任务，所以出现 VLAC、SOLE-R1、VLA-RL process reward、World-Env instant reflector、VLA-R1 verified rewards。
- 表征与动作：扩散/flow/action chunk 很难直接算 logprob 或做 TD，所以出现 CO-RFT 的 chunked critic、RL-100 的 denoising 内 PPO、RLT 的 RL token、小 actor-critic。
- 稳定性：不能让策略离开 VLA prior 太远，所以出现 RPD/VLAJS 的 teacher guidance、RLT 的 action anchor、IN-RIL 的 IL interleave、RL-100 的 offline-to-online staged recipe。

### 3. 大方向是 generalist pretrain，再转 specialist post-train

这条线很清楚：

- generalist 层：OpenVLA、pi0/pi0.5/pi0.6、LingBot、SmolVLA。
- specialist 层：RLT、RECAP、GR-RL、RL-100、CO-RFT、RPD、RM-RL。

大家默认 base VLA 提供语义、视觉和粗动作 prior；真正的可靠性、速度、精度来自后训练，尤其是失败状态上的在线经验、correction、value/progress 信号。

### 4. 奖励模型正在从“外部打分器”升级成“时序进度理解器”

旧思路是用 CLIP/VLM 给当前图像打分；新论文普遍认为这不够，因为单帧打分容易被视角、遮挡和 reward hacking 误导。

更强的做法是：

- pairwise progress delta：VLAC；
- video-native per-timestep progress：SOLE-R1；
- value/Q 作为 task progress：GR-RL；
- verified trajectory reward：VLA-RFT、VLA-R1；
- world model rollout 后再评价：World-Env、VLA-RFT、IRL-VLA。

规律：奖励不再只是“是否成功”，而是“当前状态相对目标推进了多少，是否退步，何时该停止”。

### 5. World model 有两种用法：训练环境或联合表征

这批论文里 world model 不是单一方向：

- 作为虚拟训练环境：World-Env、VLA-RFT、IRL-VLA，用来减少真实交互和安全成本。
- 作为 VLA 的联合学习目标：RynnVLA-002，让模型既会出动作，又会预测动作后果。
- 作为评测基础设施：RobotArena infinity，通过 real-to-sim 放大评测规模。

规律：world model 的吸引力在“可重置、可 rollout、可生成反事实”，但代价是模型误差和仿真可信度。

### 6. 推理链正在进入机器人，但必须和动作/视觉奖励绑定

VLA-R1、ThinkAct、SOLE-R1 都不是简单让模型多输出几句 CoT。它们都加了 grounding：

- VLA-R1：affordance region、trajectory、format verifiable rewards；
- ThinkAct：goal completion + trajectory consistency reward；
- SOLE-R1：视频中每一步发生了什么、是否推进任务；
- pi0.5：高层 subtask prediction 直接服务低层动作。

规律：机器人 CoT 的价值不在“解释得好听”，而在能否改善可执行的 affordance、trajectory、progress 和 recovery。

### 7. 评测从“我在自家环境跑通”转向标准化、可复现、跨分布

vla-eval、ManipArena、RobotArena infinity 都在解决同一个问题：VLA 论文太容易因为不同 benchmark、不同机器人、不同 reset 和不同成功判据而不可比。

未来可信结果至少要说明：

- ID/OOD 怎么分；
- reset 和初始状态怎么控制；
- 成功率以外有没有速度、恢复、鲁棒性；
- 是否在统一协议下比较多个模型；
- 是否有真实或 real-to-sim 的分布转移测试。

## 三、如果要“找规律”做方向，最值得盯的空位

1. 开源 VLA 的低成本真实后训练
   PI 的 RLT/RECAP 很强，但依赖内部 pi0.6 接口和数据。普通实验室在 SmolVLA/OpenVLA/OpenPI 上做 post-hoc RL token、residual actor、failure-stage correction 仍有空间。

2. 精密瓶颈阶段而不是全任务 RL
   RLT、GR-RL、RL-100 都说明最值钱的是“最后一厘米/最后一毫米”的 correction。把 RL 限定在 failure-prone phase，会比全程 RL 更省数据、更容易出结果。

3. Progress reward + action prior 的组合
   单独做 reward model 会撞 VLAC/SOLE-R1；单独做 actor-critic 会不稳。更合理的是用 progress critic 做辅助信号，同时用 VLA reference action/anchor 保持策略安全。

4. Action chunk / diffusion / flow 的 RL 接口
   这是很多论文的技术难点。CO-RFT、RLT、RL-100、StreamingVLA 都从不同角度处理它。如果能给开源 VLA 提供统一的 chunk-level RL adaptor，会很有价值。

5. 评测协议本身要像论文贡献的一部分
   单个 demo 不够。需要报告少样本曲线、失败模式、速度、扰动鲁棒性、OOD 初始状态，以及和 SFT/RPD/VLAJS/CO-RFT 类 baseline 的比较。

## 四、一句话总规律

这批论文的共同范式是：

**大规模 VLA 负责“会做很多事”，后训练 RL/critic/world model/reasoning 负责“在真实分布里做稳、做快、做准”。**

真正的创新点通常不在 VLA backbone 本身，而在如何把有限真实经验转化成稳定的 progress 信号、可控的策略更新，以及可复现的评测闭环。
