# AutoTask4macOS

[English README](README_en.md)

AutoTask4macOS 是一个 macOS 上的远程 GPU 监控与任务启动工具。它可以读取远程服务器的 GPU 状态，也可以把远端 `.sh` 脚本放进 `tmux` 后台运行。

## 准备

- macOS
- 本机已安装 Conda
- 本机可以通过 `ssh <Host>` 登录远程服务器
- 远程服务器已安装 NVIDIA 驱动、`nvidia-smi` 和 `tmux`

AutoTask4macOS 不保存密码，也不处理交互式密码登录；请先配置好 SSH 密钥或免密登录。

## 开始使用

1. 双击 `setup.command`，按提示完成首次环境配置。
2. 双击 `start_monitor.command` 打开 GPU 监控页。
3. 双击 `start_runner.command` 打开 Runner。

首次打开网页时，点击 **编辑服务器**，粘贴你的 SSH config，例如：

```sshconfig
Host gpu-box-01
  HostName gpu.example.com
  User yourname
  Port 22
```

这里的 `Host` 名称就是之后页面里选择服务器时看到的名字。

## Monitor

Monitor 用来看远程 GPU 状态。打开后会自动刷新一次，之后点击 **刷新** 才会再次访问远程服务器。

你可以看到 GPU 显存、使用率、功率、温度和进程信息。点击 **使用高亮** 后，会高亮 SSH config 中 `User` 对应用户的进程。

## Runner

Runner 用来启动远端任务。点击 **添加任务** 后选择服务器、显卡、Conda 环境，填写任务名和远端 sh 脚本路径。

默认脚本路径是：

```text
~/projects/autotask.sh
```

启动后会在远程服务器创建一个独立的 `tmux` session。关闭本地网页不会停止远端任务。任务列表里可以打开 tmux、终止程序、删除 tmux 或删除本地任务记录。

## 排查

- 双击 `.command` 没反应：确认文件有执行权限。
- 找不到 Conda：先安装 Anaconda 或 Miniconda，再重新双击 `setup.command`。
- SSH 失败：先在终端确认 `ssh <Host>` 可以成功登录。
- 读不到 GPU：确认远程服务器可以运行 `nvidia-smi`。
- 任务无法启动：确认远端已安装 `tmux`，并且脚本路径存在。
- 日志位置：`data/monitor.log` 和 `data/runner.log`。Monitor 和 Runner 关闭时会清空各自日志。

## License

Apache License 2.0
