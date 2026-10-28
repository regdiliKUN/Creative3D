# Creative3D · 3D 建模创意生成器

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey.svg)](https://www.microsoft.com/windows)
[![Offline](https://img.shields.io/badge/network-not%20required-brightgreen.svg)](#)

一个**完全离线**的 Windows 桌面工具：把一句话想法或一张图片，变成可 3D 打印的参数化模型。

```
"冷色的尖锐晶簇，细节 80"  ──►  🌟 实时预览  ──►  STL / OBJ / PNG / 创意卡
        (文本)                                        ▲
         或者                                          │
    [ 一张照片 / 手绘草图 ]  ──►  主色·明暗·轮廓  ─────┘
        (图像)
```

---

## ✨ 特性

### 多模态输入
- **文本**：中文自然语言解析出形态、配色、风格、密度、种子，并回显「已识别」结果便于确认。
- **图像**：拖入照片或手绘，自动提取主色板、对比度、对称性、边缘密度、亮度高度场与径向轮廓，
  用来驱动配色、浮雕起伏和旋转体外形。
- 两种输入可叠加：图片定形，文字定调。

### 8 个参数化形态族
波纹花瓶 · 晶簇尖晶 · 环面纽结 · 螺旋塔 · 起伏地块 · 螺旋海螺 · 图像浮雕 · 图像轮廓

每个形态都有 3 组调好的预设，参数 A / B 含义随形态变化，全部形态保证
**封闭实体 + 法线朝外**，可直接进切片软件。

### 实时预览
- numpy 向量化生成 + Pillow 光栅化，拖拽时低精度保流畅，松手后自动高清重绘。
- 平滑着色（顶点法线插值）、投影阴影、透明背景可选。
- 视口与导出图**共用同一套投影与着色代码**，所见即所得。

### 一键导出
二进制 STL（毫米）、OBJ（带配色）、PNG（1400×1050，2× 超采样）、Markdown 创意卡。
导出在后台线程跑，带进度条、可取消、失败自动清理半成品。

### 可靠性
- 统一异常体系 + 可展开详情的错误对话框
- 日志落盘到 `%LOCALAPPDATA%\Creative3D\logs`
- 设置文件损坏自动备份并重置
- 102 项单元/集成测试 + 内置 `--selftest` 无界面自检（校验全部形态的封闭性与法线）

---

## 🚀 使用

### 直接运行（无需安装 Python）
从 [Releases](../../releases) 下载 `3D建模创意生成器.exe`，双击即可。

### 从源码运行
```bash
pip install numpy pillow
cd src
python run.py
```

### 命令行
```bash
3D建模创意生成器.exe --selftest          # 无界面自检，逐形态校验封闭性
3D建模创意生成器.exe --make-shortcut     # 只创建桌面快捷方式
3D建模创意生成器.exe --version
```

### 快捷键

| 键 | 功能 |
| --- | --- |
| `Ctrl+Enter` | 解析输入框内容并应用 |
| `N` | 换一组随机灵感 |
| `R` / 双击 | 复位视角 |
| `空格` | 暂停 / 恢复自转 |
| `1` `2` `3` `4` | 正面 / 斜视 / 侧面 / 俯视 |
| `W` | 线框 |
| `Ctrl+Z` / `Ctrl+Shift+Z` | 撤销 / 重做 |
| `Ctrl+E` | 导出 |
| `F1` | 帮助 |

---

## 📁 项目结构

```
Creative3D/
├── 3D建模创意生成器.exe      # 单文件可执行程序（v2 构建产物）
├── 使用说明.md
├── legacy_v1/                # v1 旧包体备份
├── exports/                  # 默认导出目录
└── src/
    ├── run.py                # 源码入口
    ├── 3D建模创意生成器.spec  # PyInstaller 打包配置
    ├── assets/               # 图标
    ├── creative3d/
    │   ├── config.py         # 常量、主题、路径
    │   ├── errors.py         # 统一异常体系
    │   ├── logsetup.py       # 日志
    │   ├── mesh.py           # 网格数据结构（法线 / 封闭性 / 有向体积 / STL / OBJ）
    │   ├── noise.py          # 值噪声 / fBm / ridged
    │   ├── generators.py     # 8 个形态生成器
    │   ├── registry.py       # 形态元数据与关键词
    │   ├── imaging.py        # 图像特征提取
    │   ├── nlp.py            # 中文提示词解析
    │   ├── ideas.py          # 创意文案生成
    │   ├── render.py         # 渲染管线（预览与导出共用）
    │   ├── session.py        # 会话状态 + 撤销/重做 + 缓存
    │   ├── settings.py       # 设置持久化
    │   ├── jobs.py           # 后台任务与取消令牌
    │   ├── exporter.py       # 导出套装
    │   ├── shortcut.py       # 桌面快捷方式
    │   ├── app.py            # 应用入口
    │   └── ui/               # 主题 / 控件 / 视口 / 面板 / 对话框 / 主窗口
    ├── tests/test_core.py    # 测试套件（102 项）
    └── tools/                # 图标生成、截图、布局体检
```

---

## 🧪 测试

```bash
cd src
python tests/test_core.py        # 102 项：几何、解析、图像、导出、会话、设置
python tools/audit.py            # 界面布局体检（控件裁切 / 溢出检测）
python tools/shot.py out.png 6   # 启动并截图
```

---

## 🔧 打包

```bash
cd src
pip install pyinstaller
pyinstaller --noconfirm --clean "3D建模创意生成器.spec"
```

产物在 `src/dist/3D建模创意生成器.exe`（单文件，约 33 MB）。

---

## 📄 许可

MIT —— 见 [LICENSE](LICENSE)。
