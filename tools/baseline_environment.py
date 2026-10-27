"""Snapshot the installed dependency closure without credentials or unrelated packages."""

import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def snapshot() -> dict:
    pending = []
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (ROOT / name).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                pending.append(Requirement(line))
    seen = set()
    versions = {}
    problems = set()
    environment = default_environment()
    while pending:
        request = pending.pop()
        if request.marker and not request.marker.evaluate({**environment, "extra": ""}):
            continue
        name = canonicalize_name(request.name)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            problems.add(f"Missing dependency: {name}")
            continue
        if not request.specifier.contains(distribution.version, prereleases=True):
            problems.add(f"{request} incompatible with installed {distribution.version}")
        versions[name] = distribution.version
        extras = set(request.extras) | {""}
        key = (name, tuple(sorted(extras)))
        if key in seen:
            continue
        seen.add(key)
        for raw in distribution.requires or []:
            dependency = Requirement(raw)
            if dependency.marker and not any(dependency.marker.evaluate({**environment, "extra": extra}) for extra in extras):
                continue
            # Already evaluated in the parent's extras context.
            dependency.marker = None
            pending.append(dependency)
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "ram_capacity_bytes": None,
            "ram_note": "Hardware capacity unavailable in sandbox; peak RSS has not been measured",
            "native_test_configuration": {"OMP_NUM_THREADS": "1", "unsafe_runtime_bypass": False},
            "dependencies": dict(sorted(versions.items())), "dependency_problems": sorted(problems),
            "scope": "Installed closure for this OS/Python, not a cross-platform or fresh-install verification"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = snapshot()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    pins = "# Captured on " + report["system"] + " " + report["machine"] + " Python " + report["python"] + "\n"
    pins += "# Use with requirements-dev.txt; platform-specific, not hash-locked.\n"
    pins += "\n".join(f"{name}=={version}" for name, version in report["dependencies"].items()) + "\n"
    (args.output_dir / "constraints.txt").write_text(pins)
    print(json.dumps({"packages": len(report["dependencies"]), "dependency_problems": report["dependency_problems"]}))
    return 1 if report["dependency_problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
