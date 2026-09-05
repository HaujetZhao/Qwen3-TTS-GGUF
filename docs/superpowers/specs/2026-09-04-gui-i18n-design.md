# GUI 多语言设计（方案 A：TOML 字典 + 语义 key）

2026-09-04。方案 B（gettext）暂缓，之后另开分支对比。

## 决策

- **语言**：先做架构，中文为真源；产出 en / de 翻译（翻译由子代理完成）。
- **切换时机**：重启生效。启动时读一次字典，不做运行时热切换。
- **key**：语义 key（点分命名），每语言一份完整字典；`t()` 查不到回退返回 key 本身。
- **格式**：TOML，`tomllib`（stdlib）读取。零新依赖、零编译步骤，PyInstaller 只需把 `locales/` 加进 datas。

## 架构

新模块 `qwen3_tts_gguf/gui/i18n.py`：

- `set_lang(code)` — 读 `gui/locales/<code>.toml` 进全局 `_catalog`
- `t(key)` — 按 `.` 逐层取嵌套表，缺失回退 key
- `available_langs()` — 扫 `locales/*.toml`，供设置页下拉

字典按页面分节（`load.*` / `gen.*` / `clone.*` / `custom.*` / `design.*` / `tools.*` / `settings.*` / `log.*` / `app.*`）。

## 语言持久化

GUI 此前零持久化。新建最小配置 `~/.qwen3_tts_gui.toml`，单键 `language = "zh"`。
配置缺失默认 `zh`，不猜系统 locale。设置页加"界面语言"下拉，选中写配置并提示重启生效。

## 翻译范围

- 翻：静态控件文案 + 动态状态文案（按钮状态切换、状态栏、弹窗提示）。
- 不翻：日志面板引擎输出、异常栈、路径。

## 验收

- 机械：各语言 toml key 集合一致、无空值；GUI 各 tab 启动无 KeyError。
- 手感（用户）：切语言重启后布局不被长文本（德语约长 30%）撑爆。
