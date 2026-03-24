from __future__ import annotations

import importlib.util
import sys
from typing import Dict, List

"""
Check which third-party packages are missing for the imports actually used
in this notebook/script, and print a pip install command for only those
missing packages.

- Ignores standard library modules.
- Handles import-name vs pip-name differences.
- Treats local/custom modules (e.g. data_loader) as NOT pip-installable.
"""

# Third-party imports actually used by your code
IMPORT_TO_PIP: Dict[str, str] = {
    "numpy": "numpy",
    "tensorflow": "tensorflow",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "ucimlrepo": "ucimlrepo",
}

# Local/custom modules from your project
LOCAL_OR_CUSTOM = {"data_loader"}


def is_importable(module_name: str) -> bool:
    """Return True if the module can be imported, else False."""
    return importlib.util.find_spec(module_name) is not None


def main() -> None:
    missing_imports: List[str] = []
    custom_missing: List[str] = []

    for import_name in sorted(IMPORT_TO_PIP):
        if not is_importable(import_name):
            missing_imports.append(import_name)

    for custom_name in sorted(LOCAL_OR_CUSTOM):
        if not is_importable(custom_name):
            custom_missing.append(custom_name)

    missing_pip_pkgs = sorted({IMPORT_TO_PIP[name] for name in missing_imports})

    print("Python executable:", sys.executable)
    print()

    if custom_missing:
        print("Local/custom modules NOT found (not pip-installable):")
        for name in custom_missing:
            print(f"  - {name}")
        print()

    if not missing_pip_pkgs:
        print("All required third-party packages are already installed.")
        return

    print("Missing third-party packages detected:")
    for pkg in missing_pip_pkgs:
        print(f"  - {pkg}")
    print()

    cmd = f'"{sys.executable}" -m pip install ' + " ".join(missing_pip_pkgs)
    print("Run this command to install ONLY the missing packages:")
    print(cmd)


if __name__ == "__main__":
    main()