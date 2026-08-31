#!/usr/bin/env python3

from bootstrap.bootstrap import seed_from_path


def resolve_bootstrap_seed(path: str, cli_seed: int | None = None) -> dict[str, int | str]:
    """
    Resolve the bootstrap RNG seed.

    An explicit CLI seed takes precedence. Otherwise derive a deterministic
    seed from the ensemble path, matching the workflow's path-based bootstrap
    convention.
    """
    if cli_seed is not None:
        return {"seed": int(cli_seed), "source": "cli"}
    return {"seed": int(seed_from_path(path)), "source": "input_dir"}


__all__ = ["resolve_bootstrap_seed"]
