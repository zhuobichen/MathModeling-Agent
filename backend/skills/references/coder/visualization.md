# CoderAgent 图表质量提升参考

> 提取自 rougier/scientific-visualization-book
> 在 coder.md 基础上补充以下细节规范

## 一、Material Design 配色方案（替代简单5色）

每组10级渐变，适合多类分组、热力图、连续变量着色：

```python
MATERIAL_COLORS = {
    "blue":   ["#e3f2fd","#bbdefb","#90caf9","#64b5f6","#42a5f5","#2196f3","#1e88e5","#1976d2","#1565c0","#0d47a1"],
    "red":    ["#ffebee","#ffcdd2","#ef9a9a","#e57373","#ef5350","#f44336","#e53935","#d32f2f","#c62828","#b71c1c"],
    "green":  ["#e8f5e9","#c8e6c9","#a5d6a7","#81c784","#66bb6a","#4caf50","#43a047","#388e3c","#2e7d32","#1b5e20"],
    "orange": ["#fff3e0","#ffe0b2","#ffcc80","#ffb74d","#ffa726","#ff9800","#fb8c00","#f57c00","#ef6c00","#e65100"],
    "purple": ["#f3e5f5","#e1bee7","#ce93d8","#ba68c8","#ab47bc","#9c27b0","#8e24aa","#7b1fa2","#6a1b9a","#4a148c"],
    "teal":   ["#e0f2f1","#b2dfdb","#80cbc4","#4db6ac","#26a69a","#009688","#00897b","#00796b","#00695c","#004d40"],
    "grey":   ["#fafafa","#f5f5f5","#eeeeee","#e0e0e0","#bdbdbd","#9e9e9e","#757575","#616161","#424242","#212121"],
}
```

**使用原则**：
- 2-3 类分组：选 blue[5], orange[5], green[5]（索引5是主色，视觉权重均衡）
- 4-6 类分组：选 blue[7], orange[7], green[7], red[7], purple[7], teal[7]
- 热力图/连续变量：选 single-hue 如 blue[0]→blue[9]
- **禁止红绿同时作为主色**（色盲不可区分）→ 用蓝橙替代

## 二、中英文混排字体栈

```python
import matplotlib.pyplot as plt
import matplotlib

# 字体回退链：优先中文字体，fallback到英文
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",    # 思源黑体（最佳）
    "SimHei",              # 黑体（Windows 自带）
    "Microsoft YaHei",     # 微软雅黑
    "Arial",               # 英文 fallback
]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

# 数学公式字体（需 LaTeX 或 mathtext）
matplotlib.rcParams["mathtext.fontset"] = "stix"  # STIX 字体，接近 Times
```

## 三、去图表垃圾（Chartjunk Removal）

参考 Edward Tufte "data-ink ratio" 原则，每张图做到：

```python
# 全局设置
plt.rcParams.update({
    "axes.spines.top": False,       # 隐藏上脊线
    "axes.spines.right": False,     # 隐藏右脊线
    "axes.grid.alpha": 0.3,         # 网格半透明
    "axes.grid.linestyle": "--",    # 虚线网格
    "axes.linewidth": 1.0,          # 坐标轴细线
    "xtick.major.width": 0.8,       # 刻度线宽
    "ytick.major.width": 0.8,
    "legend.frameon": False,        # 图例无边框
    "legend.loc": "upper right",
})
```

**具体操作每张图前**：
```python
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.3)
```

**禁止**：
- 饼图、3D图、excel默认配色
- 四边完整边框
- 粗网格线、深色背景
- 超过2个子图

## 四、印刷质量输出

```python
# 论文最终用图
fig.savefig("figures/figure1.png", dpi=300, bbox_inches="tight", facecolor="white")
# 无损矢量格式（可选，用于 LaTeX）
fig.savefig("figures/figure1.pdf", bbox_inches="tight")
```
