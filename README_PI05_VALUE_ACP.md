# Pi0.5 Piper: Value + ACP + VLA Retrain

本文档只针对当前这台机器上的 `pi05 piper` 流程，目的是把后续离线 RL 迭代固定成一套可重复执行的步骤。

适用前提：

- 你已经完成了 `pi05` 的基础 VLA 训练
- 你已经完成了分机推理验证
- 当前基础策略 checkpoint 为：

```bash
/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model
```

- 当前原始数据集目录为：

```bash
/home/lenovo/Evo-RL/piper_multitask_v1
```

---

## 1. 总体流程

每一轮都按下面顺序做：

1. 冻结一份数据集快照 `D_k`
2. 在 `D_k` 上训练 value function
3. 在 `D_k` 上跑 value inference，写回 `value / advantage / indicator`
4. 用带 `indicator` 的 `D_k` 重新训练 `pi05`
5. 拿新策略去部署、收新数据，再进入下一轮

注意：

- `lerobot-value-infer` 会直接原地改数据集 parquet，不是只在 `output_dir` 里存结果
- 所以不要直接在原始数据集 `piper_multitask_v1` 上跑 value inference
- 先复制一份数据集快照，再在快照上打标签

---

## 2. 当前仓库中的对应关系

- Value training: `lerobot-value-train`
- Value inference: `lerobot-value-infer`
- ACP VLA retrain: `lerobot-train --acp.enable=true`

当前实现约束：

- `value_train` 目前只支持 `--value.type=pistar06`
- `value_train` 依赖 episode 成功标签，默认字段是 `episode_success`
- `ACP` 重训时会把 `acp_indicator` 注入到 task 文本里训练 `pi05`

---

## 3. Iteration 1

### 3.1 先复制一份数据集快照

```bash
cd /home/lenovo/Evo-RL
cp -a /home/lenovo/Evo-RL/piper_multitask_v1 /home/lenovo/Evo-RL/piper_multitask_v1_iter1
```

### 3.2 统一变量

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

export DATASET_ID=piper_multitask_v1_iter1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_multitask_v1_iter1

export BASE_POLICY=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model

export VALUE_RUN=pi05_piper_value_iter1_bs16_32k
export TAG=iter1
export POLICY_RUN=pi05_piper_acp_iter1
```

### 3.3 训练 value function

```bash
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-value-train \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.use_gradient_checkpointing=true \
  --batch_size=16 \
  --steps=32000 \
  --value.scheduler_warmup_steps=2000 \
  --value.scheduler_decay_steps=32000 \
  --output_dir=outputs/value_train/$VALUE_RUN \
  --job_name=$VALUE_RUN \
  --wandb.disable_artifact=true \
  --wandb.enable=true
```

说明：

- 这条命令是按当前这张 `32G` 卡调整过的 value 训练配置
- 原始默认配方是 `batch_size=64, steps=8000`，总样本预算是 `64 x 8000 = 512000`
- 现在改成 `batch_size=16` 后，为了保持接近的总样本预算，需要把步数同步放大到 `32000`
- `value.use_gradient_checkpointing=true` 是为了把显存压到 32G 可用范围
- `wandb.disable_artifact=true` 是为了避免训练结束后卡在 checkpoint artifact 上传
- 这一步会在 `outputs/value_train/$VALUE_RUN` 下保存 value 模型
- 后面 `value_infer` 直接使用这个目录作为 `--inference.checkpoint_path`

### 3.4 运行 value inference，并把标签写回数据集

```bash
lerobot-value-infer \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --inference.checkpoint_path=outputs/value_train/$VALUE_RUN \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_$TAG \
  --acp.advantage_field=complementary_info.advantage_$TAG \
  --acp.indicator_field=complementary_info.acp_indicator_$TAG \
  --output_dir=outputs/value_infer/$VALUE_RUN \
  --job_name=$VALUE_RUN.infer
```

这一步结束后，数据集 `$DATASET_ROOT` 里会新增三列：

```bash
complementary_info.value_iter1
complementary_info.advantage_iter1
complementary_info.acp_indicator_iter1
```

### 3.5 用 ACP 标签重训 pi05

```bash
lerobot-train \
  --dataset.repo_id=$DATASET_ID \
  --dataset.root=$DATASET_ROOT \
  --policy.type=pi05 \
  --policy.pretrained_path=$BASE_POLICY \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_$TAG \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/train/$POLICY_RUN \
  --job_name=$POLICY_RUN \
  --wandb.enable=true
```

完成后，新策略通常从下面路径取：

```bash
outputs/train/$POLICY_RUN/checkpoints/<STEP>/pretrained_model
```

如果你训练跑满并使用最后一个 checkpoint，也可以直接用最后一轮对应的 `pretrained_model`。

---

## 4. Iteration 2 及之后

下一轮重复同样流程，但一定要换：

- 数据集快照目录
- `VALUE_RUN`
- `TAG`
- `POLICY_RUN`
- `BASE_POLICY`

示例：

```bash
cp -a /home/lenovo/Evo-RL/piper_multitask_v1 /home/lenovo/Evo-RL/piper_multitask_v1_iter2

export DATASET_ID=piper_multitask_v1_iter2
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_multitask_v1_iter2

export BASE_POLICY=/home/lenovo/Evo-RL/outputs/train/pi05_piper_acp_iter1/checkpoints/<STEP>/pretrained_model

export VALUE_RUN=pi05_piper_value_iter2_bs16_32k
export TAG=iter2
export POLICY_RUN=pi05_piper_acp_iter2
```

然后继续执行：

1. `lerobot-value-train`
2. `lerobot-value-infer`
3. `lerobot-train --acp.enable=true`

---

## 5. 复现时必须保留的东西

如果你想几周后还能完全复现某一轮，至少保留以下内容：

- 数据集快照目录，例如 `piper_multitask_v1_iter1`
- value 训练输出目录，例如 `outputs/value_train/pi05_piper_value_iter1_bs16_32k`
- policy 训练输出目录，例如 `outputs/train/pi05_piper_acp_iter1`
- 当轮的 tag 名称，例如 `iter1`
- 当轮使用的基础策略路径
- 当时仓库的 git commit

建议每轮都额外记一份：

- value run 名称
- policy run 名称
- `n_step`
- `positive_ratio`
- `indicator_dropout_prob`
- 实际用于部署的 checkpoint 路径

---

## 6. 数据要求检查

在开始 value 训练前，至少确认：

- 数据集里有 `episode_success`
- 数据集里的 `task` 文本是正常的
- `episode_index / frame_index / index` 完整存在

补充：

- `complementary_info.is_intervention` 如果存在，`value_infer` 会优先把 intervention 帧标成正样本
- 如果这个字段不存在，脚本会按全 0 处理，不会报错

---

## 7. 最简执行顺序

如果只是快速开始，按下面顺序照跑即可：

```bash
cd /home/lenovo/Evo-RL
conda activate evo-rl

cp -a /home/lenovo/Evo-RL/piper_multitask_v1 /home/lenovo/Evo-RL/piper_multitask_v1_iter1

export DATASET_ID=piper_multitask_v1_iter1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_multitask_v1_iter1
export BASE_POLICY=/home/lenovo/Evo-RL/outputs/train/pi05_piper_v1_bs32_30k_0416_181219/checkpoints/030000/pretrained_model
export VALUE_RUN=pi05_piper_value_iter1_bs16_32k
export TAG=iter1
export POLICY_RUN=pi05_piper_acp_iter1
```

然后依次执行：

1. `lerobot-value-train`
2. `lerobot-value-infer`
3. `lerobot-train`

---

## 8. 当前建议

第一次做这条链时，不要同时改 RTC、异步推理框架、数据格式。

先把下面这条闭环做通：

1. `baseline dataset`
2. `value train`
3. `value infer`
4. `ACP pi05 retrain`
5. `新 checkpoint 部署验证`

等这条闭环稳定之后，再去比较：

- baseline `pi05`
- ACP `pi05`
- ACP + RTC `pi05`
