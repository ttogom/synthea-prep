"""Own an isolated local Ollama server without changing the user's GUI daemon."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_SERVER_ENVIRONMENT = {
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_NO_CLOUD": "1",
}
# Provider credentials and proxy credentials are unnecessary for local inference.
# In particular, do not pass the parent runner's loaded .env wholesale to a
# server that reports its configuration in its own log.
_INHERITED_ENVIRONMENT = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "LC_ALL",
    "OLLAMA_MODELS", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "GGML_METAL_PATH_RESOURCES",
)


def _address(host):
    if not isinstance(host, str) or not host.strip():
        raise ValueError("ollama_host must name a dedicated loopback host and port")
    parsed = urlsplit(host if "://" in host else "http://" + host)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("ollama_host must name a valid loopback port") from exc
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/") or port is None or port < 1024
            or port == 11434):
        raise ValueError("ollama_host must use a dedicated loopback port other than 11434")
    hostname = parsed.hostname
    address = f"[{hostname}]:{port}" if ":" in hostname else f"{hostname}:{port}"
    return hostname, port, address


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class ManagedOllama:
    """A private server whose process, port lock, and log have one owner.

    NUM_PARALLEL and CONTEXT_LENGTH configure the server before model loading.
    Reducing client concurrency later does not reduce its allocated slot count.
    This class never kills or adopts a server it did not start.
    """

    def __init__(self, settings, log_path, *, parallel=None, context=None, max_loaded_models=None):
        if not isinstance(settings, dict):
            raise TypeError("Ollama execution settings must be an object")
        self.enabled = settings.get("manage_ollama", True)
        if type(self.enabled) is not bool:
            raise ValueError("manage_ollama must be a boolean")
        self.hostname, self.port, self.address = _address(settings.get("ollama_host", "127.0.0.1:11435"))
        self.base_url = "http://" + self.address
        self.parallel = _positive(settings.get("ollama_parallel_slots", 4) if parallel is None else parallel,
                                  "Ollama parallel slots")
        self.context = _positive(settings.get("ollama_context_length", 8192) if context is None else context,
                                 "Ollama context length")
        self.max_loaded_models = _positive(
            settings.get("ollama_max_loaded_models", 1) if max_loaded_models is None else max_loaded_models,
            "Ollama maximum loaded models")
        self.log_path = Path(log_path).resolve()
        self._process = self._log = self._port_lock = None
        self._entered = False
        self._version = None

    def _request(self, method, path, payload=None, *, timeout=2):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(self.base_url + path, data=body, method=method,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError("Ollama returned a non-object response")
        return value

    def _port_busy(self):
        try:
            with socket.create_connection((self.hostname, self.port), timeout=0.2):
                return True
        except ConnectionRefusedError:
            return False
        except OSError as exc:
            # Connection refusal means unused; timeouts and other failures do not
            # establish ownership and must not cause us to launch over a service.
            if exc.errno in (61, 111):
                return False
            raise ValueError(f"Cannot establish that dedicated Ollama port {self.port} is unused") from exc

    def _acquire_port(self):
        path = Path(tempfile.gettempdir()) / f"medcopilot-ollama-{os.getuid()}-{self.port}.lock"
        self._port_lock = path.open("a", encoding="utf-8")
        try:
            fcntl.flock(self._port_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._port_lock.close()
            self._port_lock = None
            raise ValueError(f"Dedicated Ollama port {self.port} is owned by another experiment") from exc

    def _environment(self):
        return {**{name: os.environ[name] for name in _INHERITED_ENVIRONMENT if name in os.environ},
                **_SERVER_ENVIRONMENT, "OLLAMA_HOST": self.address,
                "OLLAMA_NUM_PARALLEL": str(self.parallel), "OLLAMA_CONTEXT_LENGTH": str(self.context),
                "OLLAMA_MAX_LOADED_MODELS": str(self.max_loaded_models)}

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Managed Ollama context is already active")
        self._entered = True
        if not self.enabled:
            return self
        try:
            self._acquire_port()
            if self._port_busy():
                raise ValueError(f"Dedicated Ollama endpoint {self.base_url} is already in use; "
                                 "refusing to reuse or stop an unowned server")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("ab")
            self._process = subprocess.Popen(
                ["ollama", "serve"], env=self._environment(), stdout=self._log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            deadline = time.monotonic() + 30
            while True:
                if self._process.poll() is not None:
                    raise RuntimeError(f"Managed Ollama exited before becoming ready; see {self.log_path}")
                try:
                    version = self._request("GET", "/api/version")
                    if not isinstance(version.get("version"), str) or not version["version"].strip():
                        raise ValueError("Ollama version response is missing its version")
                    self._version = version["version"]
                    break
                except (URLError, OSError, HTTPError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Managed Ollama was not ready within 30 seconds; see {self.log_path}")
                    time.sleep(0.1)
            return self
        except BaseException:
            self.close()
            raise

    def metadata(self):
        """Return whitelisted effective settings and installed model identities."""
        if not self.enabled:
            return {"managed": False}
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Managed Ollama server is not running")
        response = self._request("GET", "/api/tags")
        models = response.get("models")
        if not isinstance(models, list) or any(not isinstance(model, dict) for model in models):
            raise ValueError("Ollama tags response must contain a model list")
        return {
            "managed": True, "base_url": self.base_url, "version": self._version,
            "parallel_slots": self.parallel, "context_length": self.context,
            "max_loaded_models": self.max_loaded_models, "keep_alive": -1, "cloud_enabled": False,
            "models": [{key: model.get(key) for key in ("name", "digest", "size")}
                       for model in models],
        }

    def ps_snapshot(self):
        """Read live model placement/context diagnostics from this owned server."""
        if not self.enabled:
            return {"models": []}
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Managed Ollama server is not running")
        return self._request("GET", "/api/ps")

    def unload(self, model):
        """Unload a drained model using an empty request; no generation is performed."""
        if not self.enabled:
            return
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model to unload must be a nonempty tag")
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Managed Ollama server is not running")
        self._request("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False}, timeout=30)

    def close(self):
        """Stop our isolated process group and release its persistent port-lock inode."""
        try:
            if self._process is not None:
                if self._process.poll() is None:
                    # start_new_session=True makes this PID the group leader;
                    # the GPU runners it launches belong to the same owned group.
                    # Never discover or signal a process by its listening port.
                    try:
                        os.killpg(self._process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(self._process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self._process.wait(timeout=5)
                else:
                    self._process.wait()
        finally:
            self._process = None
            if self._log is not None:
                self._log.close()
                self._log = None
            if self._port_lock is not None:
                fcntl.flock(self._port_lock.fileno(), fcntl.LOCK_UN)
                self._port_lock.close()
                self._port_lock = None
            self._entered = False

    def __exit__(self, kind, value, traceback):
        self.close()


def plan_judge_settings(metadata, *, concurrency=None, context_length=None):
    """Resolve explicit settings, then the last Plan stage, then run defaults.

    Plan-stage settings can change between invocations without changing the
    generation experiment. Returning their effective values keeps the client
    executor and the separately owned Ollama server in agreement on resume.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Recorded run metadata must be an object")
    experiment = metadata.get("experiment")
    stage = metadata.get("plan_evaluation")
    experiment = {} if experiment is None else experiment
    stage = {} if stage is None else stage
    if not isinstance(experiment, dict) or not isinstance(stage, dict):
        raise ValueError("Recorded experiment and plan_evaluation must be objects")
    execution = experiment.get("execution")
    execution = {} if execution is None else execution
    if not isinstance(execution, dict):
        raise ValueError("Recorded experiment.execution must be an object")
    if concurrency is None:
        concurrency = stage.get("concurrency")
        if concurrency is None:
            concurrency = experiment.get("plan_judge_concurrency", 2)
    if context_length is None:
        context_length = stage.get("context_length")
        if context_length is None:
            context_length = execution.get("ollama_judge_context_length",
                                           execution.get("ollama_context_length", 8192))
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("Plan judge concurrency must be an integer from 1 to 64")
    if type(context_length) is not int or not 512 <= context_length <= 262144:
        raise ValueError("Plan judge context length must be an integer from 512 to 262144")
    return {"concurrency": concurrency, "context_length": context_length}


@contextmanager
def ollama_for_saved_run(run_dir, *, concurrency=None, context_length=None):
    """Use a run's recorded server settings for its separate Plan-judging phase."""
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    effective = plan_judge_settings(metadata, concurrency=concurrency, context_length=context_length)
    experiment = metadata.get("experiment") or {}
    settings = experiment.get("execution") or {}
    if not isinstance(settings, dict):
        raise ValueError("Recorded experiment.execution must be an object")
    settings = {"manage_ollama": False, **settings}
    with ManagedOllama(settings, run_dir / "ollama-judge.log",
                       parallel=effective["concurrency"], context=effective["context_length"]) as server:
        yield server
