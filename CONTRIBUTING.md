# 贡献指南（Contributing Guide）

欢迎贡献到 **King.triton-kernel**。本指南说明如何搭建开发环境、跑测试、遵循代码风格、提交 PR，以及最重要的——**评测口径纪律**。

---

## 1. 开发环境搭建

### 前置要求

- **Python ≥ 3.10**（推荐 3.10；`requires-python = ">=3.10"`）
- Linux + NVIDIA GPU（**运行 GPU 逻辑必需**；纯逻辑测试在 CPU 上也能跑）
- CUDA 驱动（`nvidia-smi` 可见即可；CUDA toolkit 由 torch/triton 自带）

### 步骤

```bash
git clone <your-fork> && cd King.triton-kernel

# 建议使用 venv
python3.10 -m venv .venv
source .venv/bin/activate

# 安装开发依赖(含 pytest/ruff 等) + 以 editable 方式安装本包
pip install -e ".[dev]"
```

> **关于重型依赖**：`torch` / `triton` 是核心依赖（`pip install -e ".[dev]"` 会按 `[project]` 的 `dependencies` 安装）。若你的机器已装好 CUDA 版 torch/triton，可跳过这部分；CI 上它们常因体积大而装不上，纯逻辑测试不依赖它们（见下文测试一节）。

训练侧如需 **verl / ray / vllm**（`drkernel/` 的 RL 集成），请单独安装 `[train]` extra，并在隔离环境中使用：

```bash
pip install -e ".[train]"
```

---

## 2. 运行测试

### 纯逻辑测试（推荐日常跑，CI 也用它）

```bash
pytest -m "not gpu and not redis and not slow"
```

- 该命令由 `pyproject.toml` 的 `addopts` 默认标记过滤：**跳过**需要真实 GPU、真实 redis、慢速/需要 eval server 的测试。
- `tests/conftest.py` 提供 `gpu_available` fixture：无 GPU 或装不上 torch 时自动 skip 对应用例，因此**纯逻辑测试在 CPU-only / CI 上也能跑**。

### 需要 GPU 的测试

```bash
# 需要真实 NVIDIA GPU, 且对应 CUDA 版 torch/triton 可用
pytest -m "gpu"
```

> ⚠️ GPU 测试会**真实编译并执行 Triton kernel**，请勿在共享/生产机器上随意跑（见 SECURITY.md 的隔离要求）。

### 冒烟

```bash
bash setup.sh
python3 smoke_test.py http://<eval-server>:8004
```

---

## 3. 代码风格（ruff）

本仓库使用 [ruff](https://docs.astral.sh/ruff/)（配置见 `pyproject.toml`，line-length=110，target py310）。提交前必须通过：

```bash
ruff check .
ruff format --check .
```

`[dev]` extra 已包含 ruff。也可在提交钩子里用 pre-commit（`[dev]` 已含 pre-commit）。

**风格要点**：

- 严格遵循 `ruff check`（select: E, F, I, W, UP, B）。
- 用 `ruff format` 格式化，不要手排对齐。
- `extend-exclude` 已排除 `drkernel/`、`docs/`、`evals/` 下的生成/文档目录，改动业务代码请勿绕开 lint。

---

## 4. PR 流程

1. **Fork + 分支**：从 `main` 开特性分支（`feat/xxx` 或 `fix/xxx`）。
2. **开发**：小步提交，提交信息清晰（可用 conventional commits 风格，如 `feat(...)` / `fix(...)` / `docs(...)`）。
3. **本地验证**（提交前必做）：
   ```bash
   ruff check .
   ruff format --check .
   pytest -m "not gpu and not redis and not slow"
   ```
   涉及 GPU 逻辑的改动请尽量额外跑对应 GPU 测试。
4. **PR checklist**：按 `.github/pull_request_template.md` 逐项勾选，特别是**评测口径纪律**一栏。
5. **CI**：PR 会自动跑 `lint` 与 `test` 两个 job，两者需通过（`test` 在装不上 torch 时也允许跳过 import-torch 模块，但纯逻辑用例仍须绿）。
6. **Review**：至少一名维护者 approve 后合并（squash）。

---

## 5. 评测口径纪律（铁律，必读）

> **本项目对评测口径极其敏感：口径混用 = 不诚实的数字。** 这是本项目的差异化主张，也是 review 的硬门槛。

任何对外引用的数字，**必须标注完整口径**，否则视为无效：

| 维度 | 必须标注 |
|------|----------|
| 指标 | sample_solve_rate / correctness / fast@1 / fast@1.2 / 其他 |
| 采样方式 | best-of N / best-turn / greedy（并注明预算） |
| 轮次范围 | 全部修复轮次 / 仅最终轮 |
| speedup 阈值 | ≥1.0x / ≥1.2x / 其他 |
| reference_cache | ON / OFF |
| 口径工具脚本 SHA | 若用到 `evals/agg_eval.py` 等口径脚本 |

**具体要求**：

- 对比不同模型/方法时，**必须用同一口径**，否则禁止直接比较。
- 修改 `evals/` 下任何口径聚合逻辑时，必须提供回归测试与前后对比说明。
- 不要"挑选"口径来让数字更好看（如 fast@1 与 fast@1.2 混着报）。
- 口径方法论细则见父仓内部文档 `eval_calibre.md`。

---

## 6. 发布清扫纪律

本仓库不包含：私有数据集、训练 checkpoint、内部日志、内部 overlay。提交代码前请自查：

- ❌ 不含绝对路径 / 内部主机名 / 内部 IP。
- ❌ 不含 `.ckpt` / `.safetensors` / 日志 dump 等大文件。
- ✅ 只提交真实代码副本（kernelgym/、drkernel/、evals/ 均为独立副本，非软链）。
- 删除文件前先列清单并确认，遵循项目硬规则。

---

## 7. 行为准则

参与本项目即视为同意 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)（Contributor Covenant 2.1）。

---

## 8. 安全

涉及**代码执行**面（评估/判分/训练侧把模型输出喂给 GPU 执行）的漏洞，请走私有披露，见 [SECURITY.md](SECURITY.md)，**不要**创建公开 issue。

---

## 问题与帮助

- 讨论功能 / 疑问 → GitHub Discussions / issue。
- Bug → 用 `.github/ISSUE_TEMPLATE/bug_report.md` 模板，务必填环境与口径。
- 功能建议 → 用 `feature_request.md` 模板。
