# VENDOR：robot_core（修复版）

真机示例运行环境所需的 `robot-core` 包，**vendor 自框架工作区
`forge_robots/packages/robot_core`（commit `9c69078`）**，仅一处修复：

`src/robot_core/arrow_validation.py` 原代码 `from forge_msgs.value import
ensure_record_batch` 引用了**不存在的模块**（forge_msgs 1.0.0 与 dev 分支均为
`forge_msgs.arrow`，该引用在两边都会 `ImportError`）；修复为
`from forge_msgs.arrow import ensure_record_batch`（同 API）。

其余文件（`__init__.py` / `node_runner.py` / `robot_protocol.py`）逐字节一致。

## 为什么 vendor

真机示例 pyproject 若依赖框架工作区的 robot-core（path 源），会同时带入其
`[tool.uv.sources]`（forge-msgs 为 meta-emt dev 源），与 kai0-inference 的
1.0.0 发布版产生双源冲突（uv 拒绝解析），且该包自身存在上述 import 缺陷。
vendor 后：依赖来源单一（根项目 override 统一到 1.0.0）、缺陷可控、客户部署自包含。

## 同步方法

框架修复上述 import（或 robot_core 版本更新）后：

```bash
cp <框架根>/forge_robots/packages/robot_core/src/robot_core/* \
   vendor/robot_core/src/robot_core/
# 重新应用 VENDOR 修复（arrow_validation.py 的 import 行），更新本文件来源 commit
```
