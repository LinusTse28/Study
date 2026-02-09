# Jupyter Notebook 快捷键完整指南

## 两种模式说明

Jupyter Notebook 有两种操作模式：

- **🔵 命令模式（Command Mode）**
  
  - 单元格边框显示为**蓝色**
  - 按 `Esc` 键进入
  - 用于操作单元格（插入、删除、移动等）

- **🟢 编辑模式（Edit Mode）**
  
  - 单元格边框显示为**绿色**
  - 按 `Enter` 键进入
  - 用于编辑单元格内容

---

## 🔵 命令模式快捷键

按 `Esc` 进入命令模式后可使用以下快捷键

### 运行单元格

| 快捷键             | 功能                     |
| --------------- | ---------------------- |
| `Shift + Enter` | **运行当前单元格，选中下一个**（最常用） |
| `Ctrl + Enter`  | 运行当前单元格，光标停留在当前        |
| `Alt + Enter`   | 运行当前单元格，在下方插入新单元格      |

### 插入和删除单元格

| 快捷键  | 功能                       |
| ---- | ------------------------ |
| `A`  | 在**上方**插入单元格（Above）      |
| `B`  | 在**下方**插入单元格（Below）⭐ 最常用 |
| `DD` | **删除**当前单元格（连按两次 D）      |
| `Z`  | **撤销**删除单元格              |

### 复制、剪切、粘贴

| 快捷键         | 功能           |
| ----------- | ------------ |
| `C`         | 复制单元格        |
| `X`         | 剪切单元格        |
| `V`         | 粘贴单元格到**下方** |
| `Shift + V` | 粘贴单元格到**上方** |

### 单元格类型切换

| 快捷键 | 功能                      |
| --- | ----------------------- |
| `Y` | 转换为**代码单元格**（Code Cell） |
| `M` | 转换为**Markdown单元格**⭐ 常用  |
| `R` | 转换为Raw单元格（纯文本）          |

### 单元格选择与移动

| 快捷键                | 功能           |
| ------------------ | ------------ |
| `↑` 或 `K`          | 选择上一个单元格     |
| `↓` 或 `J`          | 选择下一个单元格     |
| `Shift + ↑`        | 向上扩展选择多个单元格  |
| `Shift + ↓`        | 向下扩展选择多个单元格  |
| `Shift + M`        | **合并**选中的单元格 |
| `Ctrl + Shift + -` | 在光标处分割单元格    |

### 显示与折叠

| 快捷键         | 功能                |
| ----------- | ----------------- |
| `L`         | 显示/隐藏当前单元格的**行号** |
| `Shift + L` | 显示/隐藏所有单元格的行号     |
| `O`         | 折叠/展开当前单元格的**输出** |
| `Shift + O` | 折叠/展开所有单元格的输出     |

### 内核操作

| 快捷键  | 功能                    |
| ---- | --------------------- |
| `II` | **中断**内核（连按两次 I，停止运行） |
| `00` | **重启**内核（连按两次 0）      |

### 滚动与其他

| 快捷键              | 功能              |
| ---------------- | --------------- |
| `Space`          | 向下滚动            |
| `Shift + Space`  | 向上滚动            |
| `H`              | 显示**快捷键帮助**面板   |
| `P`              | 打开命令面板          |
| `S` 或 `Ctrl + S` | **保存** Notebook |

---

## 🟢 编辑模式快捷键

按 `Enter` 进入编辑模式后可使用以下快捷键

### 代码编辑

| 快捷键                             | 功能               |
| ------------------------------- | ---------------- |
| `Tab`                           | **代码补全**或缩进⭐ 最常用 |
| `Shift + Tab`                   | 显示**函数文档**（工具提示） |
| `Ctrl + ]`                      | 增加缩进             |
| `Ctrl + [`                      | 减少缩进             |
| `Ctrl + A`                      | 全选               |
| `Ctrl + Z`                      | 撤销               |
| `Ctrl + Y` 或 `Ctrl + Shift + Z` | 重做               |
| `Ctrl + /`                      | 注释/取消注释选中行       |

### 运行单元格（编辑模式也可用）

| 快捷键             | 功能                |
| --------------- | ----------------- |
| `Shift + Enter` | 运行当前单元格，选中下一个     |
| `Ctrl + Enter`  | 运行当前单元格，停留在当前     |
| `Alt + Enter`   | 运行当前单元格，在下方插入新单元格 |

### 多光标编辑

| 快捷键            | 功能         |
| -------------- | ---------- |
| `Ctrl + Click` | 添加多个光标     |
| `Alt + 鼠标拖动`   | 列选择（矩形选择）  |
| `Ctrl + D`     | 选中下一个相同的单词 |

---

## 

### 1. 快速查看函数文档

```python
# 方法一：在函数后加 ?
print?

# 方法二：在函数后加 ??（查看源码）
print??

# 方法三：输入函数名后按 Shift + Tab
print  # 光标在这里，按 Shift + Tab
```

### 2. Magic命令（魔法命令）

```python
# 测试代码运行时间
%timeit sum(range(100))

# 测试整个单元格运行时间
%%time
for i in range(1000):
    pass

# 显示当前目录
%pwd

# 列出文件
%ls

# 查看所有magic命令
%lsmagic

# 显示matplotlib图表
%matplotlib inline

# 显示变量
%who  # 简单列表
%whos  # 详细信息

# 执行Python文件
%run script.py

# 加载外部代码
%load script.py

# 清空输出
%clear
```

### 3. Shell命令

在代码前加 `!` 可以执行系统命令：

```python
!pip install numpy  # 安装包
!pip list          # 列出已安装的包
!ls                # 列出文件（Linux/Mac）
!dir               # 列出文件（Windows）
!pwd               # 显示当前路径
!python --version  # 查看Python版本
```

### 4. 显示变量和数据

```python
# 单元格最后一行会自动显示（不需要print）
df.head()  # 显示DataFrame前5行

# 显示多个结果
from IPython.display import display
display(df1)
display(df2)

# 在Markdown中嵌入变量
# 方法：使用IPython.display
from IPython.display import Markdown
name = "Alice"
Markdown(f"Hello, **{name}**!")
```

### 5. Markdown单元格高级用法

```markdown
# 数学公式（LaTeX）
行内公式：$E = mc^2$

独立公式：
$$
\int_{a}^{b} f(x)dx
$$

# 表格
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| A   | B   | C   |
| 1   | 2   | 3   |

# 代码块
```python
def hello():
    print("Hello World")
```

# 引用

> 这是一段引用文字

# 任务列表

- [x] 已完成任务
- [ ] 未完成任务

# HTML（支持部分HTML）

<font color="red">红色文字</font>

```
### 6. 调试技巧

```python
# 进入交互式调试器
%debug

# 在异常发生时自动进入调试器
%pdb on

# 在单元格开头使用，出错时自动调试
%%debug
```

---

## 📋 常用工作流程

### 典型工作流程

```
1. Esc           # 进入命令模式
2. B             # 在下方新建单元格
3. Enter         # 进入编辑模式
4. 写代码
5. Shift + Enter # 运行并跳到下一个单元格
```

### 快速调试流程

```
1. 写代码
2. Ctrl + Enter  # 运行当前单元格（不跳转）
3. 修改代码
4. Ctrl + Enter  # 再次运行（反复调试）
```

### 添加说明文字

```
1. Esc           # 进入命令模式
2. B             # 新建单元格
3. M             # 转为Markdown
4. Enter         # 进入编辑
5. 写说明文字
6. Shift + Enter # 渲染显示
```

### 批量操作单元格

```
1. Esc            # 进入命令模式
2. Shift + ↓     # 选择多个单元格
3. 选择操作：
   - Shift + M   # 合并单元格
   - DD          # 删除选中单元格
   - C → V       # 复制并粘贴
   - X → V       # 剪切并粘贴
```

### 整理Notebook

```
1. Esc           # 命令模式
2. L             # 显示行号
3. Shift + O     # 折叠所有输出
4. Ctrl + S      # 保存
```

---

## 🎯 新手必记快捷键（TOP 10）

**从这10个开始：**

1. `Shift + Enter` - **运行单元格并跳转**⭐⭐⭐
2. `B` - **在下方插入单元格**⭐⭐⭐
3. `M` - **转为Markdown单元格**⭐⭐⭐
4. `DD` - **删除单元格**⭐⭐
5. `Tab` - **代码补全**⭐⭐⭐
6. `Shift + Tab` - **查看函数文档**⭐⭐
7. `Ctrl + Enter` - **运行但不跳转**⭐⭐
8. `A` - 在上方插入单元格⭐
9. `Z` - 撤销删除⭐
10. `H` - 查看快捷键帮助⭐

**基础工作流（必会）：**

```
Esc → B → Enter → 写代码 → Shift + Enter
```

---

## 🔍 查看快捷键帮助

### 方法一：快捷键

在命令模式下按 `H`

### 方法二：菜单栏

**Help → Keyboard Shortcuts**

### 方法三：命令面板

按 `P` 打开命令面板，搜索想要的功能

---

## 🌟 Markdown单元格说明

### 什么是Markdown单元格？

- **白色/空白背景**（代码单元格是灰色背景）
- 没有 `In [ ]:` 标记
- 用于写**说明文字、标题、列表**等
- 支持Markdown和HTML语法
- **双击进入编辑**，`Shift + Enter` 渲染

### 代码单元格 vs Markdown单元格对比

| 特性    | 代码单元格      | Markdown单元格 |
| ----- | ---------- | ----------- |
| 背景颜色  | 灰色         | 白色          |
| 左侧标记  | `In [ ]:`  | 无           |
| 用途    | 运行Python代码 | 写说明文字       |
| 创建快捷键 | `Y`        | `M`         |
| 注释方式  | `#`        | Markdown语法  |

### 创建Markdown单元格

```
方法一（推荐）：
1. Esc        # 进入命令模式
2. B          # 新建单元格
3. M          # 转为Markdown
4. Enter      # 开始编辑
5. 写内容
6. Shift + Enter  # 渲染显示

方法二：
点击单元格 → 顶部下拉菜单 "Code" → 改为 "Markdown"
```

### Markdown常用语法

```markdown
# 一级标题（最大）
## 二级标题
### 三级标题
#### 四级标题

**粗体文字**
*斜体文字*
~~删除线~~

- 无序列表项1
- 无序列表项2
  - 子列表项

1. 有序列表1
2. 有序列表2

`行内代码`

```python
# 代码块
print("Hello World")
```

[超链接文字](https://example.com)

![图片描述](image.png)

> 引用文字
> 多行引用

---

分割线

数学公式：$x^2 + y^2 = z^2$

```
---

## 📝 提高效率的技巧

### 1. 使用命令模式快速操作

大部分时间在命令模式下工作，只在写代码时进入编辑模式：
- `Esc` → 命令模式 → 快速插入、删除、移动单元格
- `Enter` → 编辑模式 → 写代码
- `Shift + Enter` → 运行 → 自动回到命令模式

### 2. 善用Markdown做笔记
```

# 在学习时的结构：

1. Markdown单元格 - 解释概念
2. 代码单元格 - 示例代码
3. Markdown单元格 - 总结要点
4. 代码单元格 - 练习题
   
   ```
   
   ```

### 3. 利用Magic命令提高效率

```python
# 测试多个方案的性能
%timeit [x**2 for x in range(1000)]
%timeit list(map(lambda x: x**2, range(1000)))

# 查看内存使用
%memit my_large_list

# 分析代码性能
%prun my_function()
```

### 4. 组织代码结构

```python
# 使用Markdown创建目录结构
# 1. 数据导入和清洗
# 2. 探索性数据分析
# 3. 特征工程
# 4. 模型训练
# 5. 结果评估
```

### 5. 快捷键组合拳

```
# 快速创建多个单元格
Esc → B → B → B  # 连续创建3个

# 快速删除多个单元格
Esc → Shift + ↓ → Shift + ↓ → DD

# 快速复制结构
Esc → C → V → V → V  # 复制后连续粘贴
```

---

## ⚙️ 自定义快捷键

如果想自定义快捷键：

1. 点击菜单：**Help → Edit Keyboard Shortcuts**
2. 或者在命令模式按 `P` 打开命令面板，搜索 "keyboard"

---

## 🚀 进阶功能

### 1. 多notebook工作

```python
# 在一个notebook中运行另一个notebook
%run other_notebook.ipynb
```

### 2. 导出功能

```
File → Download as →
- Python (.py)
- HTML (.html)
- Markdown (.md)
- PDF via LaTeX (.pdf)
```

### 3. 扩展插件

安装Jupyter扩展获得更多功能：

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

常用扩展：

- **Table of Contents** - 自动生成目录
- **Code folding** - 代码折叠
- **ExecuteTime** - 显示执行时间
- **Variable Inspector** - 变量查看器

---

## 📚 学习资源

- **官方文档**：https://jupyter-notebook.readthedocs.io/
- **快捷键列表**：在Notebook中按 `H` 查看
- **Markdown语法**：https://www.markdownguide.org/
- **Magic命令**：在cell中运行 `%quickref`

---

## 🎓 小土堆PyTorch学习建议

如果你是为了学习小土堆的PyTorch教程：

### 推荐工作流程

```
1. 创建新notebook
2. 第一个单元格（Markdown）：写今天的学习目标
3. 导入库（代码单元格）
4. 每个知识点：
   - Markdown解释概念
   - 代码演示
   - Markdown记录要点
5. 最后一个单元格（Markdown）：总结今天学到的
```

### 示例结构

```markdown
# Day 1: PyTorch基础

## 学习目标
- 了解Tensor的基本操作
- 掌握自动求导机制

## 1. Tensor创建
[代码单元格]

### 要点
- 使用torch.tensor()创建
- 支持多种数据类型

## 2. Tensor运算
[代码单元格]

## 总结
今天学习了...下次要学习...
```

---

*最后更新：2024年2月*

**提示：** 在Jupyter中按 `H` 可随时查看最新的快捷键列表！
