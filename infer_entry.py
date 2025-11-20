"""Stable, minimal entrypoint delegating to `infer_clean.main()`.

Use this if `infer.py` is corrupted in the workspace. Run:

    python infer_entry.py --mode sd --prompt "your prompt..."

"""

import sys

try:
    from infer_clean import main
except Exception:
    try:
        from .infer_clean import main  # type: ignore
    except Exception as e:
        sys.stderr.write(f"Failed to import infer_clean: {e}\n")
        raise


if __name__ == "__main__":
    main()
