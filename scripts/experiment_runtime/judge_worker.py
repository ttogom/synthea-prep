"""Private child entry point; the lightweight supervisor owns server and run lock."""

import json
from pathlib import Path
import sys

from judge_treatment_plans import main
from scripts.experiment_runtime.ollama import OllamaRuntimeView


if __name__ == "__main__":
    bootstrap = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    settings = bootstrap["profile"]["settings"]
    raise SystemExit(main([bootstrap["run_dir"], "--concurrency", str(settings["concurrency"]),
                           "--context-length", str(settings["context_length"])],
                          root=Path(bootstrap["root"]), _lock_fd=bootstrap["lock_fd"],
                          _server=OllamaRuntimeView(bootstrap["server"]),
                          _stop_file=Path(bootstrap["stop_file"]), _guard_metadata=bootstrap["guard_metadata"],
                          _runtime_profile=bootstrap["profile"]))
