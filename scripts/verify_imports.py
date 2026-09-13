"""
verify_imports.py  –  Smoke-test that every src module imports cleanly.

Run from the project root after `pip install -e .`:
    python scripts/verify_imports.py
"""
import sys


def main() -> None:
    modules = [
        "src.config",
        "src.transforms",
        "src.dataset",
        "src.model",
        "src.utils",
        "src.explain",
        "src.train",
        "src.evaluate",
    ]

    ok = True
    for name in modules:
        try:
            __import__(name)
            print(f"  [OK]  {name}")
        except Exception as exc:
            print(f"  [FAIL] {name}  →  {exc}")
            ok = False

    if ok:
        print("\n✓ All modules imported successfully.")
    else:
        print("\n✗ Some imports failed. See output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
