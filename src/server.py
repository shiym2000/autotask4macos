#!/usr/bin/env python3
"""Local web server for monitoring remote NVIDIA GPUs over SSH."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.parse import parse_qs
import xml.etree.ElementTree as ET


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CONFIG_PATH = DATA_DIR / "servers.conf"
INDEX_PATH = APP_DIR / "index.html"
RUNNER_INDEX_PATH = APP_DIR / "runner.html"
MONITOR_LOG_PATH = DATA_DIR / "monitor.log"
RUNNER_LOG_PATH = DATA_DIR / "runner.log"
TASKS_PATH = DATA_DIR / "tasks.json"
REFRESH_INTERVAL = 2
SSH_CONCURRENCY_LIMIT = 1
SSH_TIMEOUT = 12
CONNECT_TIMEOUT = 5
DEFAULT_PORT = 8765
RUNNER_DEFAULT_PORT = 8775
CLIENT_TIMEOUT = 180
APP_MODE = "monitor"


@dataclass
class HostConfig:
    alias: str
    options: dict[str, str] = field(default_factory=dict)

    @property
    def hostname(self) -> str:
        return self.options.get("hostname", self.alias)

    @property
    def user(self) -> str:
        return self.options.get("user", "")

    @property
    def port(self) -> str:
        return self.options.get("port", "")

    def public_dict(self) -> dict[str, str]:
        return {
            "alias": self.alias,
            "hostname": self.hostname,
            "user": self.user,
            "port": self.port,
        }


STATE: dict[str, Any] = {
    "started_at": time.time(),
    "updated_at": None,
    "refresh_interval": REFRESH_INTERVAL,
    "hosts": [],
    "results": [],
    "config_error": None,
}
CLIENT_STATE: dict[str, Any] = {
    "seen": False,
    "last_seen": None,
}
TASKS: list[dict[str, Any]] = []
SSH_SEMAPHORE: threading.BoundedSemaphore | None = None


def ensure_data_files() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CONFIG_PATH.exists():
            CONFIG_PATH.write_text("", encoding="utf-8")
        if not TASKS_PATH.exists():
            TASKS_PATH.write_text("[]\n", encoding="utf-8")
        MONITOR_LOG_PATH.touch(exist_ok=True)
        RUNNER_LOG_PATH.touch(exist_ok=True)
    except OSError:
        pass


def current_log_path() -> Path:
    return RUNNER_LOG_PATH if APP_MODE == "runner" else MONITOR_LOG_PATH


def clear_current_log() -> None:
    try:
        ensure_data_files()
        current_log_path().write_text("", encoding="utf-8")
    except OSError:
        pass


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    try:
        ensure_data_files()
        with current_log_path().open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def stop_soon(reason: str, delay: float = 0.25) -> None:
    log(reason)

    def stop() -> None:
        clear_current_log()
        os._exit(0)

    threading.Timer(delay, stop).start()


def parse_ssh_config_text(text: str) -> list[HostConfig]:
    hosts: list[HostConfig] = []
    current_aliases: list[str] = []
    current_options: dict[str, str] = {}

    def flush_current() -> None:
        nonlocal current_aliases, current_options
        for alias in current_aliases:
            if "*" in alias or "?" in alias or alias.strip() == "":
                continue
            hosts.append(HostConfig(alias=alias, options=dict(current_options)))
        current_aliases = []
        current_options = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        key = parts[0].lower()
        value = parts[1].strip() if len(parts) > 1 else ""
        if key == "host":
            flush_current()
            current_aliases = value.split()
        elif current_aliases:
            current_options[key] = value
    flush_current()
    return hosts


def load_tasks() -> None:
    global TASKS
    ensure_data_files()
    if not TASKS_PATH.exists():
        TASKS = []
        return
    try:
        data = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        TASKS = []
        return
    TASKS = data if isinstance(data, list) else []


def save_tasks() -> None:
    try:
        ensure_data_files()
        TASKS_PATH.write_text(json.dumps(TASKS, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        log(f"Failed to save tasks: {exc}")


def sanitize_session_name(name: str) -> str:
    session = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    session = session.strip("._-")
    if not session:
        session = "task"
    return session[:48]


def make_unique_session(base: str) -> str:
    existing = {str(task.get("session", "")) for task in TASKS}
    if base not in existing:
        return base
    stamp = time.strftime("%m%d_%H%M%S")
    candidate = f"{base}_{stamp}"
    if candidate not in existing:
        return candidate
    return f"{candidate}_{int(time.time())}"


def remote_conda_command(args: str) -> str:
    return (
        "for f in \"$HOME/miniconda3/etc/profile.d/conda.sh\" "
        "\"$HOME/anaconda3/etc/profile.d/conda.sh\" "
        "\"/opt/anaconda3/etc/profile.d/conda.sh\" "
        "\"/opt/miniconda3/etc/profile.d/conda.sh\"; do "
        "[ -f \"$f\" ] && . \"$f\" && break; "
        "done; "
        f"if command -v conda >/dev/null 2>&1; then conda {args}; "
        f"elif [ -x \"$HOME/miniconda3/bin/conda\" ]; then \"$HOME/miniconda3/bin/conda\" {args}; "
        f"elif [ -x \"$HOME/anaconda3/bin/conda\" ]; then \"$HOME/anaconda3/bin/conda\" {args}; "
        "else echo \"找不到 conda\" >&2; exit 127; fi"
    )


def remote_script_arg(path: str) -> str:
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return f'"$HOME"/{shlex.quote(path[2:])}'
    return shlex.quote(path)


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return dict(task)


async def run_ssh_command(command: list[str], timeout: int = SSH_TIMEOUT) -> tuple[int, str, str]:
    async def execute() -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return 124, "", "SSH command timed out"
        return (
            proc.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    if SSH_SEMAPHORE is None:
        return await execute()
    await asyncio.to_thread(SSH_SEMAPHORE.acquire)
    try:
        return await execute()
    finally:
        SSH_SEMAPHORE.release()


def ssh_base_command() -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        f"ConnectTimeout={CONNECT_TIMEOUT}",
    ]


async def run_ssh(host_alias: str, remote_command: str, timeout: int = SSH_TIMEOUT) -> tuple[int, str, str]:
    command = [
        *ssh_base_command(),
        host_alias,
        remote_command,
    ]
    try:
        return await run_ssh_command(command, timeout=timeout)
    except FileNotFoundError:
        return 127, "", "本机找不到 ssh 命令"
    except OSError as exc:
        return 255, "", str(exc)


async def refresh_status_once(host_alias: str | None = None) -> None:
    hosts, config_error = load_hosts()
    STATE["hosts"] = [host.public_dict() for host in hosts]
    STATE["config_error"] = config_error
    selected_hosts = [host for host in hosts if not host_alias or host.alias == host_alias]
    fresh_results = await asyncio.gather(*(collect_host(host) for host in selected_hosts)) if selected_hosts else []
    if host_alias:
        fresh_aliases = {item.get("alias") for item in fresh_results}
        kept_results = [item for item in STATE["results"] if item.get("alias") not in fresh_aliases and item.get("alias") != host_alias]
        STATE["results"] = kept_results + fresh_results
    else:
        STATE["results"] = fresh_results
    STATE["updated_at"] = time.time()


def load_hosts() -> tuple[list[HostConfig], str | None]:
    ensure_data_files()
    if not CONFIG_PATH.exists():
        return [], None
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        return parse_ssh_config_text(text), None
    except OSError as exc:
        return [], f"无法读取 servers.conf: {exc}"


def parse_mib(text: str | None) -> int | None:
    if not text:
        return None
    token = text.strip().split()[0].replace(",", "")
    try:
        return int(float(token))
    except ValueError:
        return None


def parse_number(text: str | None) -> float | None:
    if not text:
        return None
    token = text.strip().split()[0].replace("%", "").replace("W", "").replace("C", "")
    try:
        return float(token)
    except ValueError:
        return None


def text_at(node: ET.Element, path: str) -> str | None:
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else None


def parse_nvidia_xml(xml_text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"nvidia-smi 返回内容无法解析: {exc}") from exc

    gpus: list[dict[str, Any]] = []
    for index, gpu in enumerate(root.findall("gpu")):
        fb = gpu.find("fb_memory_usage")
        util = gpu.find("utilization")
        power = gpu.find("power_readings")
        if power is None:
            power = gpu.find("gpu_power_readings")
        temp = gpu.find("temperature")
        processes_node = gpu.find("processes")
        processes: list[dict[str, Any]] = []

        if processes_node is not None:
            for proc in processes_node.findall("process_info"):
                used_memory = text_at(proc, "used_memory")
                processes.append(
                    {
                        "pid": text_at(proc, "pid") or "",
                        "user": text_at(proc, "process_user") or "",
                        "name": text_at(proc, "process_name") or text_at(proc, "type") or "unknown",
                        "used_memory_mib": parse_mib(used_memory),
                        "used_memory": used_memory or "",
                    }
                )

        total = parse_mib(text_at(fb, "total") if fb is not None else None)
        used = parse_mib(text_at(fb, "used") if fb is not None else None)
        free = parse_mib(text_at(fb, "free") if fb is not None else None)
        used_ratio = round((used / total) * 100, 1) if total and used is not None else None

        gpus.append(
            {
                "index": index,
                "id": gpu.attrib.get("id", str(index)),
                "name": text_at(gpu, "product_name") or "Unknown GPU",
                "memory": {
                    "total_mib": total,
                    "used_mib": used,
                    "free_mib": free,
                    "used_percent": used_ratio,
                },
                "utilization_percent": parse_number(text_at(util, "gpu_util") if util is not None else None),
                "power_watts": parse_number(text_at(power, "power_draw") if power is not None else None),
                "power_limit_watts": parse_number(text_at(power, "power_limit") if power is not None else None),
                "temperature_c": parse_number(text_at(temp, "gpu_temp") if temp is not None else None),
                "processes": processes,
            }
        )

    if not gpus:
        raise ValueError("没有在 nvidia-smi 输出中发现 GPU")
    return gpus


async def fetch_process_users(host: HostConfig, pids: list[str]) -> dict[str, str]:
    unique_pids = sorted({pid for pid in pids if pid.isdigit()}, key=int)
    if not unique_pids:
        return {}

    remote = f"ps -o pid= -o user= -p {shlex.quote(','.join(unique_pids))}"
    code, stdout, _ = await run_ssh(host.alias, remote, timeout=CONNECT_TIMEOUT)
    if code != 0:
        return {}

    users: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            users[parts[0]] = parts[1]
    return users


async def refresh_task_status(task: dict[str, Any]) -> None:
    if task.get("status") not in {"running", "stopped"}:
        return
    host = str(task.get("host", ""))
    session = str(task.get("session", ""))
    done_file = str(task.get("done_file", ""))
    if done_file:
        done_code, stdout, _ = await run_ssh(host, f"cat {shlex.quote(done_file)} 2>/dev/null", timeout=CONNECT_TIMEOUT)
        if done_code == 0:
            task["status"] = "finished"
            task["exit_code"] = stdout.strip().splitlines()[-1] if stdout.strip() else "0"
            task["updated_at"] = time.time()
            return
    code, stdout, stderr = await run_ssh(host, f"tmux has-session -t {shlex.quote(session)}", timeout=CONNECT_TIMEOUT)
    if code != 0:
        message = f"{stdout}\n{stderr}".lower()
        if "can't find session" in message or "no server running" in message:
            task["status"] = "stopped"
            task["updated_at"] = time.time()
        else:
            task["last_error"] = (stderr.strip() or stdout.strip() or "任务状态刷新失败")
        return
    else:
        capture_code, stdout, _ = await run_ssh(host, f"tmux capture-pane -pt {shlex.quote(session)} -S -80", timeout=CONNECT_TIMEOUT)
        task["status"] = "finished" if capture_code == 0 and "[autotask] 程序已结束" in stdout else "running"
    task["updated_at"] = time.time()


async def refresh_all_tasks() -> None:
    if not TASKS:
        return
    await asyncio.gather(*(refresh_task_status(task) for task in TASKS))
    save_tasks()


async def start_remote_task(payload: dict[str, Any]) -> dict[str, Any]:
    hosts, _ = load_hosts()
    aliases = {host.alias for host in hosts}
    host = str(payload.get("host", "")).strip()
    if host not in aliases:
        raise ValueError("请选择 SSH config 中已有的服务器")

    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("请输入任务名")

    gpus = [str(item).strip() for item in payload.get("gpus", []) if str(item).strip() != ""]
    if not all(gpu.isdigit() for gpu in gpus):
        raise ValueError("GPU 序号必须是数字")

    script_path = str(payload.get("script_path", "")).strip()
    if not script_path:
        raise ValueError("请输入远端 sh 脚本路径")
    conda_env = str(payload.get("conda_env", "")).strip()

    session = make_unique_session(sanitize_session_name(name))
    gpu_env = ",".join(gpus)
    done_file = f"/tmp/autotask4macos_{session}_{int(time.time())}.done"
    env_prefix = f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_env)}; " if gpu_env else ""
    script_arg = remote_script_arg(script_path)
    run_script = (
        remote_conda_command(f"run --no-capture-output -n {shlex.quote(conda_env)} bash {script_arg}")
        if conda_env
        else f"bash {script_arg}"
    )
    inner = (
        f"rm -f {shlex.quote(done_file)}; "
        f"{env_prefix}{run_script}; "
        "rc=$?; "
        f"printf '%s\\n' \"$rc\" > {shlex.quote(done_file)}; "
        "printf '\\n[autotask] 程序已结束，退出码: %s\\n' \"$rc\"; "
        "exec bash"
    )
    remote = f"tmux new-session -d -s {shlex.quote(session)} -- bash -lc {shlex.quote(inner)}"
    code, stdout, stderr = await run_ssh(host, remote, timeout=SSH_TIMEOUT)
    now = time.time()
    task = {
        "id": f"{session}-{int(now)}",
        "name": name,
        "session": session,
        "host": host,
        "gpus": gpus,
        "conda_env": conda_env,
        "script_path": script_path,
        "status": "running" if code == 0 else "failed_to_start",
        "started_at": now,
        "updated_at": now,
        "done_file": done_file,
        "exit_code": "",
        "attach_command": f"ssh -t {host} {shlex.quote('tmux attach -t ' + session)}",
        "tmux_command": f"tmux attach -t {shlex.quote(session)}",
        "last_error": (stderr.strip() or stdout.strip()) if code != 0 else "",
    }
    TASKS.append(task)
    save_tasks()
    if code != 0:
        raise RuntimeError(task["last_error"] or "远端 tmux 启动失败")
    return task


async def list_remote_conda_envs(host: str) -> list[str]:
    remote = "bash -lc " + shlex.quote(remote_conda_command("env list"))
    code, stdout, _ = await run_ssh(
        host,
        remote,
        timeout=SSH_TIMEOUT,
    )
    if code != 0:
        return []
    envs = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts:
            continue
        name = parts[1] if parts[0] == "*" and len(parts) > 1 else parts[0]
        if name and name not in envs:
            envs.append(name)
    return envs


async def delete_remote_task(task_id: str) -> dict[str, Any]:
    task = next((item for item in TASKS if str(item.get("id")) == task_id), None)
    if not task:
        raise ValueError("任务不存在")
    session = str(task.get("session", ""))
    code, stdout, stderr = await run_ssh(str(task.get("host", "")), f"tmux kill-session -t {shlex.quote(session)}", timeout=SSH_TIMEOUT)
    if code != 0 and "can't find session" not in stderr.lower():
        raise RuntimeError(stderr.strip() or stdout.strip() or "删除 tmux session 失败")
    task["status"] = "deleted"
    task["updated_at"] = time.time()
    save_tasks()
    return task


async def remove_task_record(task_id: str) -> dict[str, Any]:
    global TASKS
    task = next((item for item in TASKS if str(item.get("id")) == task_id), None)
    if not task:
        raise ValueError("任务不存在")
    await delete_remote_task(task_id)
    TASKS = [item for item in TASKS if str(item.get("id")) != task_id]
    save_tasks()
    return task


async def stop_remote_task(task_id: str) -> dict[str, Any]:
    task = next((item for item in TASKS if str(item.get("id")) == task_id), None)
    if not task:
        raise ValueError("任务不存在")
    session = str(task.get("session", ""))
    code, stdout, stderr = await run_ssh(str(task.get("host", "")), f"tmux send-keys -t {shlex.quote(session)} C-c", timeout=SSH_TIMEOUT)
    if code != 0:
        raise RuntimeError(stderr.strip() or stdout.strip() or "终止任务程序失败")
    task["status"] = "interrupted"
    task["updated_at"] = time.time()
    save_tasks()
    return task


async def capture_task_output(task_id: str) -> dict[str, str]:
    task = next((item for item in TASKS if str(item.get("id")) == task_id), None)
    if not task:
        raise ValueError("任务不存在")
    session = str(task.get("session", ""))
    remote = f"tmux capture-pane -pt {shlex.quote(session)} -S -200"
    code, stdout, stderr = await run_ssh(str(task.get("host", "")), remote, timeout=SSH_TIMEOUT)
    if code != 0:
        raise RuntimeError(stderr.strip() or stdout.strip() or "读取 tmux 内容失败")
    return {"output": stdout, "attach_command": str(task.get("attach_command", ""))}


def attach_process_users(gpus: list[dict[str, Any]], users: dict[str, str]) -> None:
    for gpu in gpus:
        for proc in gpu.get("processes", []):
            pid = str(proc.get("pid", ""))
            if users.get(pid):
                proc["user"] = users[pid]


async def collect_host(host: HostConfig) -> dict[str, Any]:
    started = time.time()
    command = [
        *ssh_base_command(),
        host.alias,
        "nvidia-smi",
        "-q",
        "-x",
    ]
    base = {
        "alias": host.alias,
        "hostname": host.hostname,
        "user": host.user,
        "port": host.port,
        "checked_at": started,
        "latency_ms": None,
    }

    try:
        code, stdout_text, stderr_text = await run_ssh_command(command, timeout=SSH_TIMEOUT)
    except TimeoutError:
        return {**base, "ok": False, "error_type": "timeout", "error": "SSH 或 nvidia-smi 响应超时"}
    except FileNotFoundError:
        return {**base, "ok": False, "error_type": "ssh_missing", "error": "本机找不到 ssh 命令"}
    except OSError as exc:
        return {**base, "ok": False, "error_type": "ssh_error", "error": str(exc)}

    latency_ms = int((time.time() - started) * 1000)
    stderr_text = stderr_text.strip()

    if code != 0:
        if code == 124:
            return {**base, "ok": False, "error_type": "timeout", "error": "SSH 或 nvidia-smi 响应超时"}
        message = stderr_text or stdout_text.strip() or f"ssh 返回退出码 {code}"
        error_type = "nvidia_smi_missing" if "nvidia-smi" in message and "not found" in message.lower() else "ssh_error"
        return {**base, "ok": False, "latency_ms": latency_ms, "error_type": error_type, "error": message}

    try:
        gpus = parse_nvidia_xml(stdout_text)
    except ValueError as exc:
        return {**base, "ok": False, "latency_ms": latency_ms, "error_type": "parse_error", "error": str(exc)}

    pids = [str(proc.get("pid", "")) for gpu in gpus for proc in gpu.get("processes", [])]
    attach_process_users(gpus, await fetch_process_users(host, pids))

    return {
        **base,
        "ok": True,
        "error_type": None,
        "error": None,
        "latency_ms": latency_ms,
        "gpus": gpus,
    }


async def heartbeat_loop() -> None:
    while True:
        if CLIENT_STATE["seen"] and CLIENT_STATE["last_seen"] is not None:
            idle_for = time.time() - CLIENT_STATE["last_seen"]
            if idle_for > CLIENT_TIMEOUT:
                stop_soon("No browser heartbeat detected; stopping AutoTask4macOS.", 0.25)
                return

        await asyncio.sleep(REFRESH_INTERVAL)


def find_free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("找不到可用本地端口")


class GpuMonitorHandler(BaseHTTPRequestHandler):
    server_version = "GpuMonitor/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        log("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, body: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/plain") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            page = RUNNER_INDEX_PATH if APP_MODE == "runner" else INDEX_PATH
            try:
                body = page.read_text(encoding="utf-8")
            except OSError:
                self.send_text(f"{page.name} 不存在", HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_text(body, content_type="text/html")
        elif path == "/runner":
            try:
                body = RUNNER_INDEX_PATH.read_text(encoding="utf-8")
            except OSError:
                self.send_text("runner.html 不存在", HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_text(body, content_type="text/html")
        elif path == "/api/config":
            hosts, config_error = load_hosts()
            self.send_json(
                {
                    "configured": bool(hosts),
                    "hosts": [host.public_dict() for host in hosts],
                    "refresh_interval": REFRESH_INTERVAL,
                    "config_error": config_error,
                    "config_text": CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else "",
                }
            )
        elif path == "/api/status":
            query = parse_qs(parsed.query)
            force_refresh = (query.get("refresh") or ["0"])[0] == "1"
            if force_refresh:
                host = (query.get("host") or [""])[0] or None
                asyncio.run(refresh_status_once(host))
            self.send_json(STATE)
        elif path == "/api/tasks":
            query = parse_qs(parsed.query)
            if (query.get("refresh") or ["0"])[0] == "1":
                try:
                    asyncio.run(refresh_all_tasks())
                except RuntimeError:
                    pass
            self.send_json({"tasks": [public_task(task) for task in TASKS]})
        elif path == "/api/remote_conda_envs":
            query = parse_qs(parsed.query)
            host = (query.get("host") or [""])[0]
            hosts, _ = load_hosts()
            aliases = {item.alias for item in hosts}
            if host not in aliases:
                self.send_json({"ok": False, "error": "请选择 SSH config 中已有的服务器"}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "envs": asyncio.run(list_remote_conda_envs(host))})
        elif path == "/api/health":
            self.send_json({"ok": True, "time": time.time()})
        else:
            self.send_text("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/heartbeat":
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            CLIENT_STATE["seen"] = True
            CLIENT_STATE["last_seen"] = time.time()
            self.send_json({"ok": True, "time": CLIENT_STATE["last_seen"]})
            return

        if path == "/api/shutdown":
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            self.send_json({"ok": True})
            stop_soon("Browser closed; stopping AutoTask4macOS.", 0.25)
            return

        if path == "/api/tasks":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                task = asyncio.run(start_remote_task(payload))
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "请求不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
                return
            except (ValueError, RuntimeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "task": public_task(task)})
            return

        if path == "/api/tasks/output":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                result = asyncio.run(capture_task_output(str(payload.get("id", ""))))
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "请求不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
                return
            except (ValueError, RuntimeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, **result})
            return

        if path == "/api/tasks/stop":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                task = asyncio.run(stop_remote_task(str(payload.get("id", ""))))
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "请求不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
                return
            except (ValueError, RuntimeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "task": public_task(task)})
            return

        if path == "/api/tasks/delete":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                task = asyncio.run(delete_remote_task(str(payload.get("id", ""))))
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "请求不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
                return
            except (ValueError, RuntimeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "task": public_task(task)})
            return

        if path == "/api/tasks/remove":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                task = asyncio.run(remove_task_record(str(payload.get("id", ""))))
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "请求不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
                return
            except (ValueError, RuntimeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "task": public_task(task)})
            return

        if path != "/api/config":
            self.send_text("Not found", HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_json({"ok": False, "error": "请求不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
            return

        text = str(payload.get("config_text", "")).strip() + "\n"
        hosts = parse_ssh_config_text(text)
        if not hosts:
            self.send_json({"ok": False, "error": "没有发现可监控的 Host，请粘贴 SSH config"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            ensure_data_files()
            CONFIG_PATH.write_text(text, encoding="utf-8")
        except OSError as exc:
            self.send_json({"ok": False, "error": f"保存失败: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        STATE["hosts"] = [host.public_dict() for host in hosts]
        self.send_json({"ok": True, "hosts": STATE["hosts"]})


def run_http_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), GpuMonitorHandler)
    log(f"AutoTask4macOS listening on http://127.0.0.1:{port}")
    server.serve_forever()
    return server


SAMPLE_XML = """<?xml version="1.0"?>
<nvidia_smi_log>
  <gpu id="00000000:01:00.0">
    <product_name>NVIDIA Test GPU</product_name>
    <fb_memory_usage>
      <total>24576 MiB</total>
      <reserved>0 MiB</reserved>
      <used>4096 MiB</used>
      <free>20480 MiB</free>
    </fb_memory_usage>
    <utilization>
      <gpu_util>35 %</gpu_util>
    </utilization>
    <temperature>
      <gpu_temp>55 C</gpu_temp>
    </temperature>
    <power_readings>
      <power_draw>120.50 W</power_draw>
      <power_limit>300.00 W</power_limit>
    </power_readings>
    <processes>
      <process_info>
        <pid>1234</pid>
        <process_name>python</process_name>
        <used_memory>1024 MiB</used_memory>
      </process_info>
    </processes>
  </gpu>
</nvidia_smi_log>
"""


def self_test() -> int:
    hosts = parse_ssh_config_text(
        """
        Host *
          ServerAliveInterval 60
        Host gpu-box-01
          HostName gpu.example.com
          User alice
          Port 22
        Host gpu-a gpu-b
          User user
        Host bad*
          User ignored
        """
    )
    assert [host.alias for host in hosts] == ["gpu-box-01", "gpu-a", "gpu-b"], hosts
    assert hosts[0].hostname == "gpu.example.com"
    assert hosts[0].user == "alice"
    assert hosts[0].port == "22"

    gpus = parse_nvidia_xml(SAMPLE_XML)
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA Test GPU"
    assert gpus[0]["memory"]["used_mib"] == 4096
    assert gpus[0]["utilization_percent"] == 35
    assert gpus[0]["power_watts"] == 120.5
    assert gpus[0]["temperature_c"] == 55
    assert gpus[0]["processes"][0]["pid"] == "1234"
    assert gpus[0]["processes"][0]["user"] == ""
    attach_process_users(gpus, {"1234": "alice"})
    assert gpus[0]["processes"][0]["user"] == "alice"
    assert sanitize_session_name(" train llama/0420 ") == "train_llama_0420"
    assert sanitize_session_name("!!!") == "task"
    print("self-test passed")
    return 0


async def amain() -> int:
    global APP_MODE, SSH_SEMAPHORE
    parser = argparse.ArgumentParser(description="AutoTask4macOS remote GPU monitor and task runner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--open", action="store_true", help="open the web UI in the default browser")
    parser.add_argument("--page", choices=["monitor", "runner"], default="monitor", help="which page to open")
    parser.add_argument("--app", choices=["monitor", "runner"], default=None, help="backend app mode")
    parser.add_argument("--self-test", action="store_true", help="run parser self tests")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    APP_MODE = args.app or args.page
    default_port = RUNNER_DEFAULT_PORT if APP_MODE == "runner" else DEFAULT_PORT
    port = find_free_port(args.port or default_port)
    load_tasks()
    loop = asyncio.get_running_loop()
    SSH_SEMAPHORE = threading.BoundedSemaphore(SSH_CONCURRENCY_LIMIT)
    loop.create_task(heartbeat_loop())
    url = f"http://127.0.0.1:{port}/runner" if args.page == "runner" else f"http://127.0.0.1:{port}/"
    if args.open:
        loop.call_later(0.8, lambda: webbrowser.open(url))
    try:
        await loop.run_in_executor(None, run_http_server, port)
    except KeyboardInterrupt:
        log("Stopped by user")
        clear_current_log()
    except Exception:
        log(traceback.format_exc())
        return 1
    return 0


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        log("Stopped by user")
        clear_current_log()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
