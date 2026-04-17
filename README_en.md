# AutoTask4macOS

[中文 README](README.md)

AutoTask4macOS is a macOS tool for remote GPU monitoring and task launching. It checks GPU status on SSH servers and can start remote `.sh` scripts inside detached `tmux` sessions.

## Requirements

- macOS
- Conda installed locally
- Local `ssh <Host>` works for your remote servers
- Remote NVIDIA driver, `nvidia-smi`, and `tmux`

AutoTask4macOS does not store passwords or handle interactive password login. Set up SSH keys or passwordless login first.

## Quick Start

1. Double-click `setup.command` and follow the first-time setup prompts.
2. Double-click `start_monitor.command` to open the GPU monitor.
3. Double-click `start_runner.command` to open Runner.

On first use, click **编辑服务器** and paste your SSH config, for example:

```sshconfig
Host gpu-box-01
  HostName gpu.example.com
  User yourname
  Port 22
```

The `Host` value is the server name shown in the app.

## Monitor

Monitor shows remote GPU status. It refreshes once when opened. Later refreshes happen only when you click **刷新**.

It shows GPU memory, utilization, power, temperature, and process details. Click **使用高亮** to highlight processes owned by the configured SSH `User`.

## Runner

Runner starts remote tasks. Click **添加任务**, then choose the server, GPU, Conda environment, task name, and remote shell script path.

The default script path is:

```text
~/projects/autotask.sh
```

After launch, AutoTask4macOS creates a detached `tmux` session on the remote server. Closing the local browser page does not stop remote tasks. The task table can open tmux, stop the program, delete tmux, or remove the local task record.

## Troubleshooting

- If a `.command` file does not open, check executable permissions.
- If Conda is missing, install Anaconda or Miniconda and rerun `setup.command`.
- If SSH fails, test `ssh <Host>` in Terminal first.
- If GPU data is missing, check that `nvidia-smi` works on the remote server.
- If a task cannot start, check that `tmux` is installed and the script path exists.
- Logs are stored at `data/monitor.log` and `data/runner.log`. Monitor and Runner clear their own logs when they close.

## License

Apache License 2.0
