# GitHub 基础操作指南

完整的Git和GitHub入门指令参考

---

## 目录
- [首次配置](#首次配置)
- [方法一：SSH连接（推荐）](#方法一ssh连接推荐)
- [方法二：HTTPS连接（简单）](#方法二https连接简单)
- [克隆仓库](#克隆仓库)
- [基本工作流程](#基本工作流程)
- [拉取更新](#拉取更新)
- [创建新仓库并推送](#创建新仓库并推送)
- [常用查看指令](#常用查看指令)
- [完整工作流程示例](#完整工作流程示例)
- [常见问题](#常见问题)

---

## 首次配置

只需配置一次，所有Git仓库都会使用这个配置：

```bash
# 配置用户名和邮箱
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"

# 查看配置
git config --list

# 查看具体配置项
git config user.name
git config user.email
```

---

## 方法一：SSH连接（推荐）

### 优点
- ✅ 一次配置，永久使用
- ✅ 不需要每次输入密码
- ✅ 更安全

### 配置步骤

#### 1. 生成SSH密钥

```bash
# 生成SSH密钥（推荐ed25519算法）
ssh-keygen -t ed25519 -C "你的邮箱@example.com"

# 如果系统不支持ed25519，使用RSA
ssh-keygen -t rsa -b 4096 -C "你的邮箱@example.com"

# 提示时一路回车即可（也可以设置密码，更安全）
```

#### 2. 复制公钥

```bash
# Mac/Linux
cat ~/.ssh/id_ed25519.pub

# Windows (PowerShell)
type $env:USERPROFILE\.ssh\id_ed25519.pub

# Windows (Git Bash)
cat ~/.ssh/id_ed25519.pub
```

复制输出的所有内容（以 `ssh-ed25519` 或 `ssh-rsa` 开头）

#### 3. 添加到GitHub

1. 登录 GitHub
2. 点击右上角头像 → **Settings**
3. 左侧菜单 → **SSH and GPG keys**
4. 点击 **New SSH key**
5. Title：随便起个名字（如"我的MacBook"）
6. Key：粘贴刚才复制的公钥
7. 点击 **Add SSH key**

#### 4. 测试连接

```bash
ssh -T git@github.com

# 成功会显示：
# Hi 你的用户名! You've successfully authenticated...
```

#### 5. 克隆仓库（SSH方式）

```bash
git clone git@github.com:用户名/仓库名.git

# 示例
git clone git@github.com:pytorch/pytorch.git
```

---

## 方法二：HTTPS连接（简单）

### 优点
- ✅ 无需配置SSH
- ✅ 适合临时使用
- ✅ 在防火墙严格的环境下更容易使用

### 缺点
- ❌ 每次push需要输入用户名和密码/token

### 配置步骤

#### 1. 创建Personal Access Token（必需）

GitHub已不再支持密码验证，需要使用Token：

1. 登录 GitHub
2. 点击右上角头像 → **Settings**
3. 左侧菜单最下方 → **Developer settings**
4. 左侧 → **Personal access tokens** → **Tokens (classic)**
5. 点击 **Generate new token** → **Generate new token (classic)**
6. 设置：
   - Note：给token起个名字
   - Expiration：选择过期时间
   - Select scopes：勾选 `repo`（完整仓库权限）
7. 点击 **Generate token**
8. ⚠️ **立即复制保存token**（只显示一次！）

#### 2. 克隆仓库（HTTPS方式）

```bash
git clone https://github.com/用户名/仓库名.git

# 示例
git clone https://github.com/pytorch/pytorch.git
```

#### 3. Push时使用Token

```bash
# 第一次push时会要求输入凭据
git push

# Username: 你的GitHub用户名
# Password: 粘贴刚才的Token（不是GitHub密码）
```

#### 4. 保存凭据（可选，避免每次输入）

```bash
# Mac
git config --global credential.helper osxkeychain

# Windows
git config --global credential.helper wincred

# Linux
git config --global credential.helper store

# 或者设置缓存（15分钟）
git config --global credential.helper cache
```

#### 5. 将现有仓库从HTTPS改为SSH

```bash
# 查看当前remote URL
git remote -v

# 修改为SSH
git remote set-url origin git@github.com:用户名/仓库名.git

# 验证修改
git remote -v
```

---

## 克隆仓库

### SSH方式（推荐）
```bash
# 克隆到当前目录
git clone git@github.com:用户名/仓库名.git

# 克隆到指定文件夹
git clone git@github.com:用户名/仓库名.git 自定义文件夹名

# 只克隆最新版本（浅克隆，节省空间和时间）
git clone --depth 1 git@github.com:用户名/仓库名.git
```

### HTTPS方式
```bash
# 克隆到当前目录
git clone https://github.com/用户名/仓库名.git

# 克隆到指定文件夹
git clone https://github.com/用户名/仓库名.git 自定义文件夹名
```

---

## 基本工作流程

### 日常操作流程

```bash
# 1. 查看当前状态
git status

# 2. 添加修改到暂存区
git add 文件名              # 添加单个文件
git add file1.txt file2.py  # 添加多个文件
git add .                   # 添加所有修改（最常用）
git add *.py                # 添加所有.py文件

# 3. 提交到本地仓库
git commit -m "提交说明：简短描述你做了什么"

# 4. 推送到GitHub
git push

# 第一次推送需要指定分支
git push -u origin main
# 之后直接 git push 即可
```

### 一步到位的组合命令

```bash
# 添加所有修改并提交
git add . && git commit -m "更新说明"

# 添加、提交、推送一气呵成
git add . && git commit -m "更新说明" && git push
```

---

## 拉取更新

```bash
# 拉取远程更新并自动合并（最常用）
git pull

# 等同于以下两步
git fetch      # 获取远程更新
git merge      # 合并到本地分支

# 拉取前查看远程有什么更新
git fetch
git log origin/main

# 拉取指定分支
git pull origin main
```

---

## 创建新仓库并推送

### 方式一：先在GitHub创建仓库，再推送本地代码

#### 1. 在GitHub网站创建新仓库
- 登录GitHub → 点击右上角 `+` → `New repository`
- 填写仓库名，选择公开/私有
- **不要勾选** "Add a README file"
- 点击 `Create repository`

#### 2. 在本地项目推送

```bash
# 进入你的项目文件夹
cd /path/to/your/project

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 第一次提交
git commit -m "Initial commit"

# 添加远程仓库（SSH方式）
git remote add origin git@github.com:你的用户名/仓库名.git

# 或使用HTTPS方式
git remote add origin https://github.com/你的用户名/仓库名.git

# 重命名分支为main（如果需要）
git branch -M main

# 推送到GitHub
git push -u origin main
```

### 方式二：在GitHub创建仓库并直接克隆

```bash
# 在GitHub创建仓库（可以勾选README）
# 然后克隆到本地
git clone git@github.com:你的用户名/仓库名.git
cd 仓库名

# 添加你的文件
# 然后正常add、commit、push
```

---

## 常用查看指令

```bash
# 查看状态（显示修改的文件）
git status

# 查看提交历史
git log
git log --oneline              # 简洁版（一行显示）
git log --graph --oneline      # 图形化显示分支
git log -5                     # 只显示最近5条

# 查看某个文件的修改历史
git log 文件名
git log -p 文件名              # 显示详细修改内容

# 查看具体修改内容
git diff                       # 查看未暂存的修改
git diff --staged              # 查看已暂存的修改
git diff HEAD                  # 查看所有修改

# 查看远程仓库信息
git remote -v                  # 查看远程仓库地址
git remote show origin         # 查看详细信息

# 查看分支
git branch                     # 本地分支
git branch -r                  # 远程分支
git branch -a                  # 所有分支

# 查看某次提交的详细信息
git show 提交ID
git show HEAD                  # 查看最新提交
```

---

## 完整工作流程示例

### 场景一：第一次使用项目

```bash
# 1. 克隆仓库
git clone git@github.com:username/my-project.git

# 2. 进入项目
cd my-project

# 3. 开始工作，修改文件...

# 4. 查看修改
git status

# 5. 添加修改
git add .

# 6. 提交
git commit -m "添加了新功能X"

# 7. 推送
git push
```

### 场景二：每天开始工作前

```bash
# 1. 进入项目目录
cd my-project

# 2. 先拉取最新代码（重要！避免冲突）
git pull

# 3. 开始工作，修改文件...

# 4. 完成后提交
git add .
git commit -m "完成了任务Y"
git push
```

### 场景三：多人协作

```bash
# 1. 工作前拉取最新代码
git pull

# 2. 修改文件...

# 3. 提交前再次拉取（防止别人刚push）
git pull

# 4. 提交并推送
git add .
git commit -m "更新说明"
git push
```

---

## 常见问题

### 1. 忘记先pull就修改了代码

```bash
# 拉取时会提示有冲突
git pull

# 手动编辑有冲突的文件，解决冲突标记
# <<<<<<< HEAD
# 你的代码
# =======
# 别人的代码
# >>>>>>> 

# 解决后
git add .
git commit -m "解决冲突"
git push
```

### 2. 想撤销刚才的修改

```bash
# 撤销未暂存的修改（还没git add）
git checkout -- 文件名
git restore 文件名            # 新版Git推荐

# 撤销已暂存的修改（已经git add）
git reset HEAD 文件名
git restore --staged 文件名   # 新版Git推荐

# 撤销最后一次提交（已经git commit但未push）
git reset --soft HEAD~1       # 保留修改
git reset --hard HEAD~1       # 丢弃修改（危险！）
```

### 3. 改错了提交信息

```bash
# 修改最后一次提交的信息（未push前）
git commit --amend -m "新的提交信息"

# 如果已经push了
git commit --amend -m "新的提交信息"
git push --force  # ⚠️ 谨慎使用，会覆盖远程历史
```

### 4. 想忽略某些文件

```bash
# 创建 .gitignore 文件
touch .gitignore

# 编辑 .gitignore，添加要忽略的文件
# 示例内容：
# *.log          # 忽略所有.log文件
# node_modules/  # 忽略整个文件夹
# .DS_Store      # Mac系统文件
# __pycache__/   # Python缓存
# *.pyc
```

### 5. 误删了文件想恢复

```bash
# 恢复被删除的文件（还没commit）
git checkout -- 文件名
git restore 文件名

# 恢复到某个历史版本
git checkout 提交ID -- 文件名
```

### 6. 查看帮助

```bash
# 查看Git命令帮助
git --help
git help clone
git clone --help

# 查看简短帮助
git clone -h
```

### 7. 配置代理（如果GitHub访问困难）

```bash
# HTTP代理
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# SOCKS5代理
git config --global http.proxy socks5://127.0.0.1:7890
git config --global https.proxy socks5://127.0.0.1:7890

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy

# 只对GitHub使用代理
git config --global http.https://github.com.proxy http://127.0.0.1:7890
```

### 8. 大文件处理

```bash
# 如果要提交大文件（>100MB），需要使用Git LFS
git lfs install
git lfs track "*.psd"  # 追踪大文件类型
git add .gitattributes
git add 大文件.psd
git commit -m "添加大文件"
git push
```

---

## 快速参考卡片

| 操作 | 命令 |
|------|------|
| 克隆仓库 | `git clone <url>` |
| 查看状态 | `git status` |
| 添加修改 | `git add .` |
| 提交 | `git commit -m "说明"` |
| 推送 | `git push` |
| 拉取 | `git pull` |
| 查看历史 | `git log --oneline` |
| 撤销修改 | `git restore 文件名` |
| 创建分支 | `git branch 分支名` |
| 切换分支 | `git checkout 分支名` |

---

## 推荐学习资源

- **官方文档**：https://git-scm.com/doc
- **GitHub文档**：https://docs.github.com
- **交互式教程**：https://learngitbranching.js.org/
- **中文教程**：https://www.liaoxuefeng.com/wiki/896043488029600

---

## 总结

**新手必记三步曲：**

```bash
git add .
git commit -m "说明"
git push
```

**每天工作前：**
```bash
git pull  # 先拉取！
```

**推荐使用SSH方式**，一次配置终身受益！

---

*最后更新：2024*
*有问题欢迎提Issue*
