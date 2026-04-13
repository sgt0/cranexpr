from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

# Map x86-64 microarchitecture levels to their Rust target-cpu value.
X86_64_LEVELS = {
    "x86-64": None,
    "x86-64-v2": "v2",
    "x86-64-v3": "v3",
    # "x86-64-v4": "v4",
}

# Map OS to the shared library extension.
LIB_EXTENSIONS = {
    "Linux": "so",
    "Windows": "dll",
    "Darwin": "dylib",
}

# Map OS to the shared library prefix.
LIB_PREFIXES = {
    "Linux": "lib",
    "Windows": "",
    "Darwin": "lib",
}


class CustomHook(BuildHookInterface[Any]):
    target_dir = Path("vapoursynth/plugins/cranexpr")

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"

        os_name = platform.system()
        arch = platform.machine()
        ext = LIB_EXTENSIONS[os_name]
        prefix = LIB_PREFIXES[os_name]
        crate_name = "cranexpr"
        lib_filename = f"{prefix}{crate_name}.{ext}"

        self.target_dir.mkdir(parents=True, exist_ok=True)

        is_x86_64 = arch in {"x86_64", "AMD64"}
        levels = X86_64_LEVELS if is_x86_64 else {"native": None}

        for target_cpu, level_suffix in levels.items():
            env = dict(__import__("os").environ)
            env["RUSTFLAGS"] = f"-C target-cpu={target_cpu}"

            cmd = ["cargo", "build", "--release"]

            subprocess.run(cmd, check=True, env=env)

            built = Path("target") / "release" / lib_filename

            dest_name = f"{prefix}{crate_name}.{level_suffix}.{ext}" if level_suffix else lib_filename

            shutil.copy2(built, self.target_dir / dest_name)

        manifest = self.target_dir / "manifest.vs"
        manifest.write_text(
            f"[VapourSynth Manifest V1]\n{prefix}{crate_name}\n",
            encoding="utf-8",
        )

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        shutil.rmtree(self.target_dir.parent, ignore_errors=True)
