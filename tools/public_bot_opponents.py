"""Build private pinned benchmark bundles for publicly readable event entrants.

The upstream forks have no explicit license. This file contains only our
loader/build code and provenance; upstream source and weights stay outside
the public repository. Runtime bundles never download or start HTTP servers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROFILES = {
    "armaan-primary": {
        "owner": "ArmaanMediratta",
        "commit": "183342c61239b9df5db22e6fb2cf426d5e2df758",
        "files": ["agent.py", "bot/__init__.py", "bot/equity.py", "bot/fasteval.py",
                  "bot/mirror.py", "bot/opponent.py", "bot/policy.py", "bot/preflop.py",
                  "bot/safety.py", "bot/timebank.py", "bot/data/preflop.npz"],
        "notes": "Actual primary callbacks; original wall-time sample budgeting retained; mirror detector default remains off.",
    },
    "dushyant-byte-me": {
        "owner": "Dushyant-Betala",
        "commit": "ba64c267aa2e8c442e5e5f824539c400faea7a64",
        "files": ["agent.py", "brain.py", "bot_data/tables.npz", "bot_data/policies.npz"],
        "notes": "Actual Blueprint callbacks and tables; original pace projection assumes 1000 hands even in short screens.",
    },
    "malki-equitynet": {
        "owner": "malki-aal",
        "commit": "70ddc9555b7016898d2b9478e2f67590cc61ffaf",
        "files": ["agent.py", "equity.py", "strategy.py", "equity_weights.npz"],
        "notes": "Actual default ML callbacks and committed NumPy neural weights, not rules-only mode.",
    },
}


ADAPTER = r'''"""Private frozen public-entrant adapter; see manifest.json for provenance."""
import builtins
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import uuid

BASE = Path(__file__).resolve().parent
META = json.loads((BASE / "manifest.json").read_text())
_API = None
_PREFIX = None
_FINDER = None
_ORIGINAL_IMPORT = builtins.__import__


class _Loader(importlib.abc.Loader):
    def __init__(self, finder, relative):
        self.finder = finder
        self.relative = relative

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = BASE / self.relative
        source = path.read_bytes()
        assert hashlib.sha256(source).hexdigest() == META["files_sha256"][self.relative]
        text = source.decode()
        if self.relative == "agent.py":
            marker = "# ===== SERVER PLUMBING"
            assert marker in text
            text = text.split(marker, 1)[0]
        local_builtins = dict(vars(builtins))
        local_builtins["__import__"] = self.finder.scoped_import
        module.__dict__["__builtins__"] = local_builtins
        exec(compile(text, str(path), "exec"), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, prefix):
        self.prefix = prefix
        self.local_roots = {Path(name).parts[0].removesuffix(".py")
                            for name in META["files_sha256"] if name.endswith(".py")}

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix + "."):
            return None
        relative = fullname[len(self.prefix) + 1:].replace(".", "/")
        for candidate, package in ((relative + "/__init__.py", True), (relative + ".py", False)):
            if candidate in META["files_sha256"]:
                return importlib.util.spec_from_file_location(
                    fullname, BASE / candidate, loader=_Loader(self, candidate),
                    submodule_search_locations=[str(BASE / relative)] if package else None)
        raise ModuleNotFoundError("Module absent from audited public bundle: " + fullname)

    def scoped_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in self.local_roots:
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        fullname = self.prefix + "." + name
        module = importlib.import_module(fullname)
        if fromlist:
            for item in fromlist:
                if item != "*" and not hasattr(module, item) and hasattr(module, "__path__"):
                    importlib.import_module(fullname + "." + item)
            return module
        return sys.modules[self.prefix + "." + name.split(".")[0]]


def _cleanup():
    global _API, _PREFIX, _FINDER
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    if _PREFIX:
        for name in list(sys.modules):
            if name == _PREFIX or name.startswith(_PREFIX + "."):
                del sys.modules[name]
    _API = _PREFIX = _FINDER = None


def _load():
    global _API, _PREFIX, _FINDER
    if _API is not None:
        return _API
    for name, digest in META["files_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, name
    _PREFIX = "_public_entrant_" + uuid.uuid4().hex
    package = ModuleType(_PREFIX)
    package.__path__ = [str(BASE)]
    package.__package__ = _PREFIX
    sys.modules[_PREFIX] = package
    _FINDER = _Finder(_PREFIX)
    sys.meta_path.insert(0, _FINDER)
    try:
        _API = importlib.import_module(_PREFIX + ".agent")
    except BaseException:
        _cleanup()
        raise
    return _API


def _check_errors(memory):
    errors = memory.get("errors", 0)
    if errors or memory.get("fallbacks", 0):
        raise RuntimeError("Public entrant internally failed: " + repr(errors))


def new_half(config, memory):
    _cleanup()
    api = _load()
    api.new_half(config, memory)
    # Fix only the seed of existing RNGs, preserving upstream decision logic.
    if META["profile"] == "armaan-primary":
        import numpy as np
        memory["np_rng"] = np.random.default_rng(640193)
    elif META["profile"] == "dushyant-byte-me":
        memory["brain"].rng.seed(640193)
    memory["_public_expected_hands"] = int(config.num_hands)
    _check_errors(memory)


def choose_action(obs, memory):
    action = _load().choose_action(obs, memory)
    _check_errors(memory)
    if action not in obs.legal_actions:
        raise RuntimeError("Public entrant returned illegal action")
    return action


def on_hand_end(result, memory):
    _load().on_hand_end(result, memory)
    _check_errors(memory)
    if result.hand_index + 1 >= memory["_public_expected_hands"]:
        _cleanup()
'''


def build(profile: str, checkout_root: Path, output_root: Path) -> Path:
    spec = PROFILES[profile]
    checkout = checkout_root / spec["owner"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != spec["commit"]:
        raise ValueError(f"Pinned checkout mismatch: {checkout}")
    official = {}
    project = Path(__file__).resolve().parents[1]
    for name in ("game.py", "evaluator.py"):
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=checkout)
        if committed != (project / name).read_bytes():
            raise ValueError(f"Public entrant has different official rules: {checkout / name}")
        official[name] = hashlib.sha256(committed).hexdigest()
    files = {}
    for name in spec["files"]:
        # Check working bytes against the immutable commit, not merely HEAD.
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=checkout)
        data = (checkout / name).read_bytes()
        if data != committed:
            raise ValueError(f"Modified public source: {checkout / name}")
        files[name] = data
    destination = output_root / profile
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    metadata = {
        "profile": profile,
        "repository": f"https://github.com/{spec['owner']}/heads-up-bundle",
        "commit": revision,
        "license": "No explicit license located; private benchmark only, do not redistribute upstream source or weights.",
        "source_audit": "No runtime outgoing network/shell/eval/pickle or actual hidden-deal access found in selected callback dependencies.",
        "adapter_changes": ["Omit HTTP server plumbing; call unchanged strategy callbacks.",
                            "Isolated import namespace; share identical official game/evaluator API.",
                            "Seed existing RNGs with 640193 for reproducibility; preserve original wall-time budgeting.",
                            "Escalate swallowed internal errors for audit instead of silently benchmarking a broken fallback."],
        "limitations": spec["notes"],
        "shared_official_sha256": official,
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "opponent.py").write_text(ADAPTER)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkouts", type=Path, default=Path("/tmp/poker-public-research"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/poker-overnight-private/public-bundles"))
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    args = parser.parse_args()
    for profile in args.profiles:
        print(build(profile, args.checkouts, args.output), flush=True)
