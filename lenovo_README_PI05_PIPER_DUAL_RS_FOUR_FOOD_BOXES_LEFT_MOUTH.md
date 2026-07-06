# [lenovo] Pi0.5 PiPER Dual RealSense Four Food Boxes to Left Mouth Notes

本文档记录当前这台机器上采集和训练这个任务：

```text
桌上放四盒食物：桂圆、莲子、红枣、花生。根据不同 task，从对应盒子里夹起一个食物，送到左边的嘴巴目标里。
```

这版先采 `100` 条 pilot 数据：四个任务各 `25` 条。推荐训练一个多任务模型，不要先拆成四个独立模型。

## 1. 当前硬件映射

仓库路径：

```bash
/home/lenovo/Evo-RL
```

Conda 环境：

```bash
conda activate evo-rl
```

当前实测 RealSense：

- `wrist`: `409122274629`, D405
- `front`: `352122272924`, D405

当前常用 CAN 约定：

- leader arm / 主臂: `can0`
- follower arm / 从臂: `can1`
- camera keys: `observation.images.wrist`, `observation.images.front`

注意：如果 `can1` 扫不到 follower 的 8 个电机，不要继续用 `robot.port=can1`。先用第 4 节确认哪个 CAN 口正常。

## 2. 场景设计

桌面放四个盒子，每个盒子里有多个同类食物：

- 桂圆盒
- 莲子盒
- 红枣盒
- 花生盒

盒子位置可以基本固定，只做小范围扰动。嘴巴目标在左侧，也只做小范围扰动。

建议固定一个布局，例如：

```text
左上: 桂圆盒       右上: 莲子盒
左下: 红枣盒       右下: 花生盒

左侧: 嘴巴目标
```

如果你实际摆放不同，按实际摆放来，但整轮 pilot 里不要频繁换大布局。每轮采完可以让盒子整体有 `1-3 cm` 小扰动，嘴巴目标有 `2-5 cm` 小扰动。

第一轮不要直接对真人嘴巴采集。先用假嘴、杯口、模型嘴或固定开口目标。

## 3. 四个任务文本

采集、训练、推理必须使用完全一致的英文 task text：

```bash
export TASK_LONGAN="Pick up one dried longan from the longan box and feed it into the mouth target on the left"
export TASK_LOTUS="Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"
export TASK_JUJUBE="Pick up one red jujube from the red jujube box and feed it into the mouth target on the left"
export TASK_PEANUT="Pick up one peanut from the peanut box and feed it into the mouth target on the left"
```

同一个数据集内用这四条 task text 区分任务。不要把四个食物分成四个 dataset，否则后面训练一个多任务模型会更麻烦。

## 4. 采集前检查

### 4.1 关闭 RealSense Viewer

```bash
pkill -f realsense-viewer || true
```

### 4.2 激活环境

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
```

### 4.3 检查 CAN

```bash
lerobot-setup-can --mode=test --interfaces=can0,can1
```

判断规则：

- teleop 采集需要 leader 和 follower 都正常。
- `ROBOT_PORT` 对应 follower，必须能扫到 `8/8`。
- `TELEOP_PORT` 对应 leader，也应该能正常读写。
- 如果只有一个口正常，先不要做人机 teleop 采集。

如果 CAN 没有 UP，可手动配置：

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

sudo ip link set can1 down || true
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
```

### 4.4 检查两个 RealSense

```bash
rs-enumerate-devices
```

```bash
lerobot-find-cameras realsense
```

直接按采集配置读帧：

```bash
python -c 'from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig; cams={"wrist":"409122274629","front":"352122272924"}; opened=[];
try:
    for name, serial in cams.items():
        cam=RealSenseCamera(RealSenseCameraConfig(serial_number_or_name=serial,width=640,height=480,fps=30,warmup_s=1)); cam.connect(); img=cam.read(); print("{} {} shape={} dtype={}".format(name, serial, getattr(img, "shape", None), getattr(img, "dtype", None))); opened.append(cam)
finally:
    [c.disconnect() for c in opened]'
```

预期输出包含：

```text
wrist 409122274629 shape=(480, 640, 3) dtype=uint8
front 352122272924 shape=(480, 640, 3) dtype=uint8
```

## 5. 可选 teleop 检查

如果 CAN 检查确认 `can1` 是 follower、`can0` 是 leader：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export TELEOP_PORT=can0

lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --display_data=true
```

如果主从臂端口和这里相反，只改 `ROBOT_PORT` 和 `TELEOP_PORT`。

## 6. 100 条 pilot 数据采集规划

目标总数：

```text
桂圆 25 条
莲子 25 条
红枣 25 条
花生 25 条
总计 100 条
```

推荐采集顺序不是一次采完某一种的 25 条，而是分 `5` 轮：

```text
第 1 轮: 桂圆 5，莲子 5，红枣 5，花生 5
第 2 轮: 桂圆 5，莲子 5，红枣 5，花生 5
第 3 轮: 桂圆 5，莲子 5，红枣 5，花生 5
第 4 轮: 桂圆 5，莲子 5，红枣 5，花生 5
第 5 轮: 桂圆 5，莲子 5，红枣 5，花生 5
```

每轮之间做很小的变化：

- 轻微移动嘴巴目标。
- 轻微移动盒子整体位置。
- 搅动每个盒子里的食物堆，让可夹取点不同。
- 保持相机、光照、盒子大布局基本一致。

每条 episode 的标准流程：

1. 从 home 或安全起始位开始。
2. 移动到 task 对应盒子上方。
3. 从盒子里夹起一个食物。
4. 稳定抬起，不拖拽其他食物。
5. 移动到左侧嘴巴目标前。
6. 放入嘴巴目标。
7. 后撤或回到安全位。

这 100 条 pilot 只采成功样本。夹错盒子、夹错食物、掉落、碰翻盒子、没有送到嘴巴目标，都用 `Left Arrow` 丢弃重录。

## 7. 双 RealSense 采集命令

先设置公共变量和采集函数：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export TELEOP_PORT=can0
export DATASET_ID=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v1

export TASK_LONGAN="Pick up one dried longan from the longan box and feed it into the mouth target on the left"
export TASK_LOTUS="Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"
export TASK_JUJUBE="Pick up one red jujube from the red jujube box and feed it into the mouth target on the left"
export TASK_PEANUT="Pick up one peanut from the peanut box and feed it into the mouth target on the left"

record_food () {
  local task_text="$1"
  local num_episodes="${2:-5}"
  local resume_mode="${3:-resume}"
  local extra_args=()

  if [ "$resume_mode" = "resume" ]; then
    extra_args=(--resume=true)
  fi

  lerobot-record \
    --robot.type=piper_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id=my_piper_follower \
    --robot.require_calibration=false \
    --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
    --teleop.type=piper_leader \
    --teleop.port="$TELEOP_PORT" \
    --teleop.id=my_piper_leader \
    --teleop.require_calibration=false \
    --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
    --dataset.repo_id="$DATASET_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.single_task="$task_text" \
    --dataset.num_episodes="$num_episodes" \
    --dataset.episode_time_s=90 \
    --dataset.reset_time_s=3 \
    --dataset.push_to_hub=false \
    --display_data=true \
    --play_sounds=false \
    "${extra_args[@]}"
}
```

第一次创建这个数据集时，第一条命令不要加 `--resume=true`，所以用 `new`：

```bash
# 第 1 轮，共 20 条
record_food "$TASK_LONGAN" 5 new
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume
```

第 2-5 轮都用 `resume`：

```bash
# 第 2 轮，共 20 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume

# 第 3 轮，共 20 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume

# 第 4 轮，共 20 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume

# 第 5 轮，共 20 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume
```

如果你想省事，也可以一次采完每类 25 条：

```bash
record_food "$TASK_LONGAN" 25 new
record_food "$TASK_LOTUS" 25 resume
record_food "$TASK_JUJUBE" 25 resume
record_food "$TASK_PEANUT" 25 resume
```

但更推荐 `5` 轮交替采集。

如果之前已经创建过同名半成品数据集，不要对半成品盲目 `resume`。更稳的做法是把 `DATASET_ID` 和 `DATASET_ROOT` 改成 `_v2`。

## 7.1 v3 补采数据：每个任务新增 20 条

当前成功率低，主要问题是盒子里多个小物体导致抓取目标不唯一，以及嘴巴目标位置有小幅变化。下一轮补采不要只随机多采，建议做一个新的 `v3` 数据集，在 `v2` 基础上每个任务追加 `20` 条，总共新增 `80` 条。

不要直接改 `v2`，先复制一份：

```bash
cd /home/lenovo/Evo-RL
rsync -a --info=progress2 \
  /home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v2/ \
  /home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3/
```

然后把采集变量切到 `v3`：

```bash
export DATASET_ID=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3
```

这轮补采的场景约束：

```text
每个盒子只放 3-5 个食物，不要一盒很多个。
每条示教固定抓“最靠近机器人/最前方/最明显”的那个，不要随机挑。
嘴巴目标加明显视觉标记，比如红圈、黑圈、杯口、漏斗或高对比开口。
每条从 home/open gripper 开始，路径尽量一致。
失败、夹错、掉落、没放进嘴巴，都用 Left Arrow 丢弃重录。
```

每个任务新增 `20` 条，建议按嘴巴目标位置分布采：

```text
中心位置 5 条
左偏 5 条
右偏 5 条
前后轻微偏移 5 条
```

四个任务总共新增 `80` 条：

```bash
# 第 1 轮：嘴巴中心位置，每个任务 5 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume

# 第 2 轮：嘴巴左偏一点，每个任务 5 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume

# 第 3 轮：嘴巴右偏一点，每个任务 5 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume

# 第 4 轮：嘴巴前后轻微偏移，每个任务 5 条
record_food "$TASK_LONGAN" 5 resume
record_food "$TASK_LOTUS" 5 resume
record_food "$TASK_JUJUBE" 5 resume
record_food "$TASK_PEANUT" 5 resume
```

补采完成后检查：

```bash
HF_DATASETS_CACHE=/tmp/hf/datasets lerobot-dataset-report \
  --dataset /home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3
```

预期：

```text
总 episode 数约 181 条
四个 task 每个新增 20 条
图像 key 仍然包含 observation.images.wrist 和 observation.images.front
```

## 7.2 v4 针对性补采：每个任务新增 25 条

v3 训练到 `30000` 后成功率仍然较低，下一轮不要只随机增加数据。先根据 eval 记录区分失败发生在选盒子、夹取、运输还是送入嘴巴，然后重点补最常见的失败阶段。

建议在 v3 基础上，每个任务新增 `25` 条，共新增 `100` 条。补采完成后，每个任务约 `70` 条，总数据约 `281` 条。

不要直接修改 v3，先复制成 v4：

```bash
cd /home/lenovo/Evo-RL
rsync -a --info=progress2 \
  /home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3/ \
  /home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4/
```

切换采集变量：

```bash
export DATASET_ID=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4
```

### 每个任务一次采集 25 条

每个任务直接一次采集 `25` 条，不需要分多次启动：

```bash
record_food "$TASK_LONGAN" 25 resume
record_food "$TASK_LOTUS" 25 resume
record_food "$TASK_JUJUBE" 25 resume
record_food "$TASK_PEANUT" 25 resume
```

每次连续采集 `25` 条时，在同一轮内按下面顺序调整场景：

```text
前 10 条：强化盒内夹取。盒内放 3-5 个物体，夹取最明显的物体。
中间 10 条：嘴巴目标做左、右、前、后、中心的小范围变化，每种位置约 2 条。
最后 5 条：针对当前 eval 最常见的失败类型补采。
```

所有示教保持统一轨迹：

```text
到物体上方 -> 垂直下降 -> 夹紧 -> 垂直抬升 -> 移动到嘴巴前方预送入点 -> 缓慢直线送入。
夹空、夹错、碰倒盒子、夹起后掉落、没有送入嘴巴的 episode 全部丢弃。
不要保留明显犹豫、来回修正或动作跳变的数据。
```

补采完成后检查：

```bash
HF_DATASETS_CACHE=/tmp/hf/datasets lerobot-dataset-report \
  --dataset /home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4
```

预期：

```text
总 episode 数约 281 条
longan 约 70 条
lotus 约 70 条
jujube 约 71 条
peanut 约 70 条
图像 key 仍然包含 observation.images.wrist 和 observation.images.front
```

## 7.3 莲子单任务补采 50 条

已经从 v5 里单独抽出了当前所有莲子旧数据，形成 `v1`：

```text
dataset:
/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_v1

当前内容:
70 episodes
30975 frames
1 task
```

昨天已经在 `v1` 上追加了 `50` 条莲子，得到 `120` 条。为了只用最近两天同一分布的数据训练，已经单独抽出昨天晚上的数据：保留原始 `episode 70` 和 `episode 72-119`，只删除坏的 `episode 71`，生成 recent 版数据集：

```text
dataset:
/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1

当前内容:
49 episodes
14320 frames
1 task
```

今天继续基于 `recent_v1` 追加 `50` 条莲子。因为数据集已经存在，必须使用 `--resume=true`。不要再往 `v1` 或旧的 `v2` 里继续采。

任务文本必须保持完全一致：

```text
Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left
```

采集命令：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export TELEOP_PORT=can0
export DATASET_ID=local/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1
export DATASET_ROOT=/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1
export TASK_LOTUS="Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"

lerobot-record \
  --robot.type=piper_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=piper_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=my_piper_leader \
  --teleop.require_calibration=false \
  --teleop.gravity_comp_tx_ratio=[1,1,1,1,1,1] \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK_LOTUS" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=3 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --resume=true
```

采集时保留小范围位置变化。夹空、掉落、没有送到嘴巴、人工手进入画面修正、明显撞乱盒子的 episode 直接丢弃重录。

采完后检查：

```bash
HF_DATASETS_CACHE=/tmp/hf/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
lerobot-dataset-report \
  --dataset /home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1
```

预期：

```text
总 episode 数约 99 条
total_tasks = 1
task text 只有莲子任务
图像 key 包含 observation.images.wrist 和 observation.images.front
```

## 8. 采集按键

普通 `lerobot-record` 采集时：

- `Right Arrow`: 提前结束当前 episode，并保存这一条。
- `Left Arrow`: 提前结束当前 episode，丢弃这一条并重录。
- `Esc`: 结束整轮采集，会保存已经完成的 episode。

这版 pilot 只保留成功样本。错夹、掉落、碰翻盒子、送错嘴巴目标，都按 `Left Arrow` 重录。

## 9. 采集后检查数据集

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

HF_DATASETS_CACHE=/tmp/hf/datasets lerobot-dataset-report \
  --dataset /home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v1
```

检查项：

- episode 数量应为 `100` 左右。本轮实测是 `101` 条。
- task 数量应为 `4`。
- 四个 task 每个应约 `25` 条。
- 图像 key 应包含：

```text
observation.images.wrist
observation.images.front
```

如果只有 `observation.images.wrist`，说明采集命令没有真正带上第二个相机，不能用于双相机训练。

## 10. pilot 训练命令

100 条 pilot 数据先训练 `15000` 步，看模型是否能区分四个盒子并稳定夹取。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_15k_pilot100_from_base

CUDA_VISIBLE_DEVICES=0 HF_DATASETS_CACHE=/tmp/hf/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v1 \
  --dataset.root=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v1 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=15000 \
  --log_freq=100 \
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false
```

如果 `15000` 步已经能基本完成，可以再补数据到每类 `50-70` 条后训 `30000` 步。不要一开始就用旧 spoon 模型 resume，这个任务是夹取小物体，不是舀饭。

当前已经完成的本机 pilot 训练：

```text
run:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_15k_pilot101_from_base_20260603_1921

final checkpoint:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_15k_pilot101_from_base_20260603_1921/checkpoints/015000/pretrained_model

log:
/home/lenovo/Evo-RL/outputs/train/logs/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_15k_pilot101_from_base_20260603_1921.log
```

训练最后状态：

```text
step:15K loss:0.078
Checkpoint policy after step 15000
End of training
```

### 10.1 v3 补采后推荐训练命令

v3 已完成补采：

```text
dataset:
/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3

rows:
74972

episodes:
181

task episode counts:
task_index 0: 45
task_index 1: 45
task_index 2: 46
task_index 3: 45
```

推荐从 `lerobot/pi05_base` 重新训练，不要默认从 v2 的 `025000` 继续训。v3 改了数据分布，重新从 base 训更干净。训练先跑到 `30000`，但 eval 时不要只看最终 `030000`，重点比较 `015000`、`020000`、`025000`、`030000`。

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_four_food_boxes_left_mouth_dual_rs_v3_bs32_30k_from_base_$(date +%Y%m%d_%H%M)
LOG=/home/lenovo/Evo-RL/outputs/train/logs/${RUN}.log
mkdir -p /home/lenovo/Evo-RL/outputs/train/logs
exec > "$LOG" 2>&1

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/tmp/hf/datasets \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3 \
  --dataset.root=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false 2>&1 | tee "$LOG"
```

如果要后台跑：

```bash
nohup bash -lc '
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_four_food_boxes_left_mouth_dual_rs_v3_bs32_30k_from_base_$(date +%Y%m%d_%H%M)
LOG=/home/lenovo/Evo-RL/outputs/train/logs/${RUN}.log
mkdir -p /home/lenovo/Evo-RL/outputs/train/logs

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/tmp/hf/datasets \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3 \
  --dataset.root=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v3 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false
' > /home/lenovo/Evo-RL/outputs/train/logs/v3_train_launcher.log 2>&1 &
```

训练后 eval 推荐顺序：

```text
先测 020000 或 025000。
如果动作明显抖，回退 015000。
如果 025000 稳但成功率还差，再测 030000。
```

训练完成后 eval 时，把第 11 节里的 `POLICY_PATH` 改到对应 v3 checkpoint，例如：

```bash
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/<RUN>/checkpoints/025000/pretrained_model
```

### 10.2 v4 针对性补采后训练命令

v4 新增了大量针对夹取和嘴巴位置的示教，数据分布发生变化。推荐从 `lerobot/pi05_base` 重新训练 `30000` 步，不要从 v3 的 `030000` 继续训练。

前台训练命令：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_four_food_boxes_left_mouth_dual_rs_v4_bs32_30k_from_base_$(date +%Y%m%d_%H%M)
LOG=/home/lenovo/Evo-RL/outputs/train/logs/${RUN}.log
mkdir -p /home/lenovo/Evo-RL/outputs/train/logs

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/tmp/hf/datasets \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4 \
  --dataset.root=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false 2>&1 | tee "$LOG"
```

后台训练命令：

```bash
setsid bash -lc '
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_four_food_boxes_left_mouth_dual_rs_v4_bs32_30k_from_base_$(date +%Y%m%d_%H%M)
LOG=/home/lenovo/Evo-RL/outputs/train/logs/${RUN}.log
mkdir -p /home/lenovo/Evo-RL/outputs/train/logs
exec > "$LOG" 2>&1

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/tmp/hf/datasets \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=local/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4 \
  --dataset.root=/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false
' </dev/null >/tmp/evo_rl_v4_train_launcher.log 2>&1 &
```

训练完成后，优先比较：

```text
020000：检查是否已经学会稳定夹取和送入。
025000：优先作为真机 eval 候选。
030000：检查成功率是否继续提升，以及是否出现抖动或过拟合。
```

当前本机 v4 训练：

```text
run:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_v4_bs32_30k_from_base_20260610_2138

dataset:
/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v4

优先测试:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_v4_bs32_30k_from_base_20260610_2138/checkpoints/025000/pretrained_model
```

等 `025000` 完整保存后再启动真机 eval。检查命令：

```bash
V4_25K=/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_v4_bs32_30k_from_base_20260610_2138/checkpoints/025000/pretrained_model

test -f "$V4_25K/config.json" && test -f "$V4_25K/model.safetensors" \
  && echo "v4 025000 is ready" \
  || echo "v4 025000 is not ready"
```

注意：后续已经生成了只过滤原始 episode `75` 的 v5 数据集，但当前这个 `025000` 是在 v4 上训练得到的。测试当前模型时不要把它写成 v5 模型。

### 10.2.1 莲子 recent 单任务 30K 训练命令

这个模型只用于莲子单任务。这里不混旧的 70 条数据，只使用昨天晚上单独抽出的 49 条 recent 数据，再加今天补采的 50 条。

训练使用的数据集是：

```text
/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1
```

recent 版当前数据量是 `49` 条。今天如果按 `7.3` 继续补采 `50` 条，训练时仍然使用同一个 `recent_v1` 数据集，数据量预期约 `99` 条。推荐从 `lerobot/pi05_base` 重新训练 `30000` 步，不要从旧莲子模型或四任务模型继续训。

前台训练命令：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_lotus_seed_recent_v1_bs32_30k_from_base_$(date +%Y%m%d_%H%M)
LOG=/home/lenovo/Evo-RL/outputs/train/logs/${RUN}.log
mkdir -p /home/lenovo/Evo-RL/outputs/train/logs

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/tmp/hf/datasets \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=local/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1 \
  --dataset.root=/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false 2>&1 | tee "$LOG"
```

后台训练命令：

```bash
setsid bash -lc '
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

RUN=pi05_piper_lotus_seed_recent_v1_bs32_30k_from_base_$(date +%Y%m%d_%H%M)
LOG=/home/lenovo/Evo-RL/outputs/train/logs/${RUN}.log
mkdir -p /home/lenovo/Evo-RL/outputs/train/logs
exec > "$LOG" 2>&1

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/tmp/hf/datasets \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=local/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1 \
  --dataset.root=/home/lenovo/Evo-RL/piper_lotus_seed_pick_to_left_mouth_dual_rs_recent_v1 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=2500 \
  --output_dir=/home/lenovo/Evo-RL/outputs/train/$RUN \
  --job_name=$RUN \
  --wandb.enable=false
' </dev/null >/tmp/evo_rl_lotus_seed_train_launcher.log 2>&1 &
```

eval 推荐先测：

```text
015000：看单任务是否已经稳定抓取和送到嘴。
025000：优先真机候选。
030000：如果 025000 不抖，再看成功率是否继续提升。
```

当前正在跑的本机训练：

```text
run:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026

025000 policy:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026/checkpoints/025000/pretrained_model
```

### 10.3 历史 pilot 15k 模型

这个模型训练在原始 `v1` 数据上，真机表现相对平滑，可作为保底回退模型。

```text
policy path:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_15k_pilot101_from_base_20260603_1921/checkpoints/015000/pretrained_model
```

### 10.4 上一版 v2 本机训练模型

这版基于修掉 episode 54 的 `v2` 数据，从 `lerobot/pi05_base` 重新训练，不是从旧 25k/30k 模型继续训。

```text
run:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_v2_from_base_20260607_0132

dataset:
/home/lenovo/Evo-RL/piper_four_food_boxes_pick_to_left_mouth_dual_rs_v2

resume source:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_v2_from_base_20260607_0132/checkpoints/022500

latest complete checkpoint:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_v2_from_base_20260607_0132/checkpoints/025000/pretrained_model
```

训练在 `2026-06-08` 手动停止，最后日志到了 `26K` 多一点，但没有保存到 `027500`，所以当前完整可用 checkpoint 是 `025000`。

关键日志：

```text
step:25K loss:0.039-0.041
Checkpoint policy after step 25000
step:26K loss:0.040
Terminated
```

eval 推荐顺序：

```text
1. 先测 025000：当前最新完整 checkpoint。
2. 如果 025000 抖动明显，回退 020000。
3. 如果 020000 仍抖，回退 015000。
```

可选 policy path：

```bash
# 推荐先测
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_v3_bs32_30k_from_base_20260609_0158/checkpoints/015000/pretrained_model

# 抖动明显时回退
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_v2_from_base_20260607_0132/checkpoints/020000/pretrained_model

# 保底回退
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_v2_from_base_20260607_0132/checkpoints/015000/pretrained_model
```

判断标准优先级：

```text
不抖 > 方向对 > 能夹取 > 能送到嘴
```

这批数据里还有夹爪/动作跳变，loss 更低不一定代表真机更稳。不要只因为 `025000` loss 更低就默认它最好。

### 10.5 历史从桌面复制进来的 25k/30k 模型

这两个是历史备份模型，模型文件本身检查没有 NaN/Inf，但之前真机推理出现过明显抖动。当前 eval 不建议默认使用它们。

历史已复制到 Evo-RL 的 `25k` checkpoint：

```text
source:
/home/lenovo/桌面/025000

checkpoint root:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_pilot101_from_base_20260604_1617/checkpoints/025000

inference policy path:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_pilot101_from_base_20260604_1617/checkpoints/025000/pretrained_model
```

这个 checkpoint 已检查过：`training_step=25000`，模型权重、normalizer、unnormalizer、optimizer state 都能正常打开，未发现 NaN/Inf。

当前已复制到 Evo-RL 的 `30k` checkpoint：

```text
source:
/home/lenovo/桌面/030000

checkpoint root:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_pilot101_from_base_20260604_1617/checkpoints/030000

inference policy path:
/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_bs32_30k_pilot101_from_base_20260604_1617/checkpoints/030000/pretrained_model
```

这个 checkpoint 已检查过：`training_step=30000`，模型权重、normalizer、unnormalizer 都能正常打开，未发现 NaN/Inf。

## 11. 正常单独推理命令

先用假嘴或固定目标跑 policy-only，不要直接对真人嘴巴跑。每个任务先跑 `5` 条。

当前代码已经修复 policy-only 连续 eval 的 reset loop 问题：episode 之间没有 teleop/policy action source 时，`--reset_to_zero_action=true` 会发送 reset action，而不是把 `None` 传给 action processor。默认 reset action 是 zero action；如果同时设置 `--reset_gripper_pos=60`，则手臂关节为 `0`，夹爪为 `60`。

注意：默认情况下第一个 episode 不会自动 reset，会直接从你启动命令时的机器人当前位置开始。这里加 `--reset_before_first_episode=true`，让第一个 episode 前也执行同样的 reset，所以 episode 0 和后续 episode 的起始状态一致。也就是说这里可以一次连续跑 `5` 条，不需要每个 episode 都重启一次。

如果仍然看到下面这个错误：

```text
ValueError: Action should be a RobotAction type (dict) got <class 'NoneType'>
```

先确认命令是在 `/home/lenovo/Evo-RL` 这个 repo 里运行，并且不要去掉 `--reset_to_zero_action=true`。

当前四食物数据的 episode 开头，夹爪大多是打开状态，`action.gripper.pos` 常见在 `50-70`。如果 reset 时把夹爪也设成 `0`，下一条 episode 开始时 policy 可能先等待/打开夹爪，表现为机械臂停住不动。因此 eval reset 使用：

```text
joint_1..joint_6 = 0
gripper.pos = 60
```

也就是手臂回零位，但夹爪保持打开。

### 11.1 莲子单任务 025000 eval

当前要测试的是莲子单任务 `025000`：

```text
/home/lenovo/Evo-RL/outputs/train/pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026/checkpoints/025000/pretrained_model
```

这个 checkpoint 已经在 `2026-06-12 15:10` 完整保存。当前训练如果还在继续，先停训练再 eval，否则 GPU 会被训练占住：

```bash
pkill -f "pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026"

pgrep -af "pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026" \
  || echo "training stopped"
```

莲子单任务 eval 命令：

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_lotus_seed_left_mouth_dual_rs_v2_bs32_30k_from_base_20260611_2026/checkpoints/025000/pretrained_model
export TASK_LOTUS="Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"
export EVAL_TAG=$(date +%Y%m%d_%H%M%S)
export DATASET_ID=local/eval_lotus_seed_025000_${EVAL_TAG}
export DATASET_ROOT=/home/lenovo/Evo-RL/eval_lotus_seed_025000_${EVAL_TAG}

test -f "$POLICY_PATH/config.json" && test -f "$POLICY_PATH/model.safetensors" \
  && echo "policy ok" || { echo "policy path bad or incomplete"; exit 1; }

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
lerobot-record \
  --robot.type=piper_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=my_piper_follower \
  --robot.require_calibration=false \
  --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK_LOTUS" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=10 \
  --reset_to_zero_action=true \
  --reset_gripper_pos=60 \
  --reset_before_first_episode=true \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --policy.device=cuda \
  --policy.path="$POLICY_PATH"
```

首次创建这个 `eval_lotus_seed_025000_...` 数据集时不要加 `--resume=true`。如果同一个 eval 数据集后面还想继续追加，再加 `--resume=true`。

### 11.2 四任务 eval 命令

```bash
cd /home/lenovo/Evo-RL
source /home/lenovo/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

export ROBOT_PORT=can1
export POLICY_PATH=/home/lenovo/Evo-RL/outputs/train/pi05_piper_four_food_boxes_left_mouth_dual_rs_v4_bs32_30k_from_base_20260610_2138/checkpoints/025000/pretrained_model
export EVAL_TAG=$(date +%Y%m%d_%H%M%S)

test -f "$POLICY_PATH/config.json" && test -f "$POLICY_PATH/model.safetensors" \
  && echo "policy ok" || { echo "policy path bad or incomplete"; exit 1; }

eval_food () {
  local food="$1"
  local task_text="$2"

  export DATASET_ID=local/eval_four_food_boxes_${food}_${EVAL_TAG}
  export DATASET_ROOT=/home/lenovo/Evo-RL/eval_four_food_boxes_${food}_${EVAL_TAG}

  printf 'POLICY_PATH=<%s>\nDATASET_ID=<%s>\nDATASET_ROOT=<%s>\nTASK_TEXT=<%s>\n' \
    "$POLICY_PATH" "$DATASET_ID" "$DATASET_ROOT" "$task_text"

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  lerobot-record \
    --robot.type=piper_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id=my_piper_follower \
    --robot.require_calibration=false \
    --robot.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "409122274629", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "352122272924", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
    --dataset.repo_id="$DATASET_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.single_task="$task_text" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=90 \
    --dataset.reset_time_s=10 \
    --reset_to_zero_action=true \
    --reset_gripper_pos=60 \
    --reset_before_first_episode=true \
    --dataset.push_to_hub=false \
    --display_data=true \
    --play_sounds=false \
    --policy.device=cuda \
    --policy.path="$POLICY_PATH"
}
```

首次创建 `eval_` 数据集时不要加 `--resume=true`。只有继续追加同一个 eval 数据集时再加。

四个任务分别执行，不能只看一个任务：

```bash
eval_food longan "Pick up one dried longan from the longan box and feed it into the mouth target on the left"
eval_food lotus "Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"
eval_food jujube "Pick up one red jujube from the red jujube box and feed it into the mouth target on the left"
eval_food peanut "Pick up one peanut from the peanut box and feed it into the mouth target on the left"
```

## 12. 评估记录表

每个任务单独统计：

```text
任务:
总次数:
选对盒子:
成功夹起一个:
送到嘴巴目标:
碰到其他盒子/食物:
掉落:
备注:
```

失败分析规则：

- 如果经常选错盒子：补四个盒子同时出现、不同 task 的对比数据。
- 如果经常夹不起来：只补对应食物的夹取数据。
- 如果送到嘴巴失败：补嘴巴目标小范围扰动的数据。
- 如果动作太快或过冲：优先用人在环或减速，不要直接对真人嘴巴跑。

## 13. 关键注意事项

- 先用假嘴或固定开口目标，不要直接对真人嘴巴做第一轮 policy-only demo。
- 这版任务没有嘴巴开闭判断，目标默认是可喂入状态。
- 四个 task text 必须固定，不要中途改词。
- 四个盒子都要一直出现在画面里，不能只保留目标盒子。
- 盒子位置可以小范围扰动，但不要大幅换布局。
- 每个盒子内部食物堆要有变化，否则模型只会记住固定夹取点。
- 采集、训练、推理的相机 key 必须保持 `wrist` 和 `front`。
- 第一次新建数据集不要加 `--resume=true`；后续追加才加。
eval_food lotus "Pick up one lotus seed from the lotus seed box and feed it into the mouth target on the left"
