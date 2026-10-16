# 修正后 PPC 与 Lee PC 的复现说明

本文档对应当前仓库中的 `geometry.py`、`PPC.py`、`PC.py`、`RANSAC.py`、
`MemorySAC.py`、`PerpAdapt.py`、`simulation.py` 和 `evaluate_corrected.py`。

## 1. 本次修正了什么

1. 每条三维圆轨迹先用 SVD/PCA 拟合所在平面，再在该平面内拟合二维圆；
   不再把三维圆误写成球面拟合。
2. 旋转轴写成“圆心（轴线上一点）+ 圆平面法向量”，不再用
   `圆心 / ||圆心||`，因此世界坐标系原点可以任意选择，只要采集期间保持固定。
3. PPC 使用两条轴线的公共垂线，并取公共垂线在 C-arm 轴上的端点
   `c_c,0`；旧实现中二元方程的符号也一并修正。
4. Lee PC 对三个 C-arm 圆心继续拟合一个圆，使用正确的点到平面正交投影，
   再分别求三个 C-arm 轴上的枢轴点。
5. `t` 由相应的 C-arm 轨迹逐帧反变换后取平均：

   ```text
   t_i = R_i^T (c_c - p_i),     t_hat = mean(t_i)
   ```

6. 固定角 `alpha=[0, 90, -90]`、`beta=[0, 45, -45]` 明确按“度”保存，
   在生成器中仅用 `np.deg2rad` 转换一次。每条轨迹内部的 `0..2π` 扫描角本来
   就是弧度，不再转换。
7. RANSAC 和 MemorySAC 的三点最小模型增加无量纲近似共线判定。退化三点会被
   丢弃并重抽，不计入有效迭代次数；圆心改为三点所在平面内的解析外心。

## 2. 六条轨迹与三种方法

设每条轨迹有 `N=100` 帧：

| 数据/方法 | 使用的轨迹 |
| --- | --- |
| 完整实例 | 3 条固定 alpha、扫描 beta 的轨迹 + 3 条固定 beta、扫描 alpha 的轨迹，共 6 条、600 帧 |
| PPC | 第 1 条 alpha 轨迹 + 第 1 条 beta 轨迹，共 2 条、200 帧 |
| Lee PC(4) | 3 条 alpha 轨迹 + 第 1 条 beta 轨迹，共 4 条、400 帧 |
| Lee PC(6) | 全部 6 条轨迹、600 帧 |

这里的“实例”是一套随机几何参数和随机传感器安装姿态下的完整仿真标定实验。
`--instances 1000` 表示独立生成 1000 套实验，而不是把一条实验重复 1000 次。

## 3. 环境安装

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. 生成无噪声数据并运行三种方法

```bash
python simulation.py \
  --output outputs/clean_seed20261015.npz \
  --instances 1000 \
  --samples-per-circle 100 \
  --seed 20261015 \
  --noise-std 0

python evaluate_corrected.py \
  --dataset outputs/clean_seed20261015.npz \
  --sac None \
  --json outputs/clean_metrics.json
```

当前版本在 Python/NumPy/SciPy 的常见 64 位环境中实测结果为：

| 方法 | `c_c,0` 平均欧氏误差 | `t` 分量 MSE |
| --- | ---: | ---: |
| PPC | `3.475185485970e-16` | `1.312327836244e-30` |
| Lee(4) | `4.162675650336e-16` | `1.199848397160e-29` |
| Lee(6) | `3.976877213303e-16` | `1.200488619813e-29` |

这与直接在内存中使用旋转矩阵时得到的
`1.368e-30 / 1.200e-29 / 1.200e-29` 同属双精度浮点舍入误差量级。
`.npz` 格式按原项目保存四元数，评估时再恢复旋转矩阵，因此不同 BLAS、NumPy、
SciPy 或 CPU 上最后几位可以不同；判断标准应是误差约为 `10^-15`（长度）和
`10^-30`（长度平方），而不是逐位相同。

若只想快速检查命令和数据格式，可追加 `--limit 10`：

```bash
python evaluate_corrected.py --dataset outputs/clean_seed20261015.npz --limit 10
```

## 5. 生成仅位置有高斯噪声的数据

下例对每一帧世界坐标平移向量的三个分量加入独立的 1 mm 高斯噪声，旋转四元数、
真实 `t` 和真实枢轴点保持不变；不加入结构化离群点，也不运行 RANSAC/MemorySAC。

```bash
python simulation.py \
  --output outputs/noise_1mm_seed20261015.npz \
  --instances 1000 \
  --samples-per-circle 100 \
  --seed 20261015 \
  --noise-std 0.001 \
  --noise-seed 20261016

python evaluate_corrected.py \
  --dataset outputs/noise_1mm_seed20261015.npz \
  --sac None \
  --json outputs/noise_1mm_metrics.json
```

仿真长度单位为米，因此 `0.001 m = 1 mm`。该命令的参考结果为：

| 方法 | `c_c,0` 平均误差（mm） | `t` 分量 MAE（mm） | `t` 分量 MSE（m²） | `t` 向量平均 L2 误差（mm） |
| --- | ---: | ---: | ---: | ---: |
| PPC | `0.239046` | `0.101764` | `1.676564e-08` | `0.203490` |
| Lee(4) | `0.217061` | `0.048324` | `3.669129e-09` | `0.096970` |
| Lee(6) | `0.208839` | `0.047518` | `3.561406e-09` | `0.095473` |

这里 Lee(4)/Lee(6) 在稠密、零均值高斯噪声下优于两轨迹 PPC，主要因为它们使用
更多轨迹和更多样本做平均；这不是离群点鲁棒性的比较。

要改变噪声，只需修改 `--noise-std`，例如：

```bash
# 0.1 mm
python simulation.py --output outputs/noise_0p1mm.npz --instances 1000 --seed 20261015 --noise-std 0.0001 --noise-seed 20261016

# 5 mm
python simulation.py --output outputs/noise_5mm.npz --instances 1000 --seed 20261015 --noise-std 0.005 --noise-seed 20261016
```

## 6. 数据格式

生成文件保持原项目的主要字段，并增加真值与元数据：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `alpha_R` | `(M, 3N, 4)` | 3 条固定 alpha、扫描 beta 轨迹的四元数，顺序为 `(x,y,z,w)` |
| `alpha_t` | `(M, 3N, 3)` | 上述姿态的世界坐标平移向量 |
| `beta_R` | `(M, 3N, 4)` | 3 条固定 beta、扫描 alpha 轨迹的四元数 |
| `beta_t` | `(M, 3N, 3)` | 上述姿态的世界坐标平移向量 |
| `t` | `(M, 3)` | 待标定的、传感器坐标系下固定偏移真值 |
| `cc` | `(M, 3)` | 世界坐标系下第 1 个 C-arm 枢轴点真值 `c_c,0` |
| `cc_all` | `(M, 3, 3)` | 三个 alpha 基准角对应的枢轴点真值 |
| `samples_per_circle` | 标量 | 每条轨迹帧数 `N` |
| `alpha_base_deg` / `beta_base_deg` | `(3,)` | 固定基准角，单位为度 |
| `noise_std` | 标量 | 平移噪声标准差，单位为米 |

`M=1000, N=100` 时，`alpha_R` 的形状是 `(1000,300,4)`；其中 300 是
三条轨迹各 100 帧，并不表示一条 300 帧的轨迹。

## 7. 指标定义

对第 `k` 个实例，真实值为 `t_k=(t_kx,t_ky,t_kz)`，估计值为 `t_hat_k`：

```text
t component MAE = mean over k and xyz of |t_hat - t|
t component MSE = mean over k and xyz of (t_hat - t)^2
t mean L2       = mean over k of ||t_hat_k - t_k||_2
c_c,0 error     = mean over k of ||c_hat_c,0,k - c_c,0,k||_2
```

`R^2` 分别对 x/y/z 三个输出计算后做等权平均。无噪声结果达到机器精度时，
`R^2` 显示为 1。

## 8. RANSAC / MemorySAC 的三点退化保护

两种共识方法使用同一个三点质量：

```text
q = ||(p1-p0) x (p2-p0)|| / max(||pi-pj||^2)
```

默认 `q < 0.01` 的三点被视为近似共线，并重新抽样。`q` 对平移、旋转和整体
尺度不变，因此换世界坐标系原点或把米改成毫米不会改变判定。可通过命令行设置：

```bash
python PerpAdapt.py \
  --dataset outputs/clean_seed20261015.npz \
  --method PPC \
  --sac RANSAC \
  --min-triangle-quality 0.01 \
  --sac-seed 20261017
```

将 `--sac RANSAC` 改成 `--sac MemorySAC` 即可选择 MemorySAC；改成
`--sac None` 则跳过内点筛选。完整参数和三个方法的组合示例见
[运行命令.md](运行命令.md)。

若要让 PPC、Lee(4)、Lee(6) 在一次评价中全部采用同一种内点筛选方法，可直接给
评价脚本传参：

```bash
python evaluate_corrected.py --dataset outputs/clean_seed20261015.npz --sac RANSAC
```

## 9. 回归测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：无噪声机器精度、固定角单位、仅平移加噪且真值不变、不经过世界原点的
两条旋转轴公共垂线、近似共线三点拒绝与重抽，以及命令行 SAC 选项。
