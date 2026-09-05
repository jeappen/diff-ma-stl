"""Verify that gcbfplus no longer depends on the d4rl package.

d4rl pins Python <3.11 and drags mujoco-py + dm_control. GCBF+ loads its
offline data from plain HDF5 trajectory files instead, so the import must
remain absent and the requirements files / Docker artifacts must not list it.
"""

import importlib
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class _BlockD4rl:
    """Context manager that makes ``import d4rl`` raise ImportError.

    We clear any cached d4rl entry from ``sys.modules`` and insert a finder
    that refuses to resolve it. This lets us exercise the gcbfplus modules
    exactly as they would run on a machine where d4rl is not installed.
    """

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "d4rl" or name.startswith("d4rl."):
                raise ImportError(f"d4rl import blocked by test: {name}")
            return None

    def __enter__(self):
        self._saved = {k: v for k, v in sys.modules.items() if k == "d4rl" or k.startswith("d4rl.")}
        for k in list(self._saved):
            del sys.modules[k]
        self._blocker = self._Blocker()
        sys.meta_path.insert(0, self._blocker)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.meta_path.remove(self._blocker)
        sys.modules.update(self._saved)
        return False


def _reimport(module_name: str):
    """Drop ``module_name`` from sys.modules (if present) and import it fresh."""
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class TestNoD4rlImports(unittest.TestCase):
    """The diffusion/environments modules must import without d4rl installed."""

    MODULES = [
        "gcbfplus.diffusion.environments.dataset",
        "gcbfplus.diffusion.environments.create_dataset_helpers",
        "gcbfplus.diffusion",
    ]

    def test_modules_import_without_d4rl(self):
        with _BlockD4rl():
            for mod in self.MODULES:
                with self.subTest(module=mod):
                    m = _reimport(mod)
                    self.assertIsNotNone(m)


class TestNoD4rlInRequirements(unittest.TestCase):
    """Active requirements files must not list d4rl as a dependency."""

    REQ_FILES = [
        "requirements.txt",
        "requirements-docker.txt",
    ]

    def test_no_d4rl_entry(self):
        for rel in self.REQ_FILES:
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            with self.subTest(file=rel):
                for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    self.assertNotRegex(
                        stripped,
                        r"(?i)(^|[\s=<>!@])d4rl([\s=<>!@]|$)",
                        msg=f"{rel}:{lineno} still declares a d4rl dependency: {line!r}",
                    )


class TestNoD4rlInDockerArtifacts(unittest.TestCase):
    """Dockerfile and helper scripts must not pip-install d4rl."""

    ARTIFACTS = [
        "Dockerfile",
        "scripts/patch_vcs_deps.sh",
        "scripts/test_docker_image.sh",
    ]

    def test_no_d4rl_install_command(self):
        for rel in self.ARTIFACTS:
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            with self.subTest(file=rel):
                text = path.read_text()
                self.assertNotIn(
                    "Farama-Foundation/d4rl",
                    text,
                    msg=f"{rel} still references the d4rl git repo",
                )
                self.assertNotIn(
                    "import d4rl",
                    text,
                    msg=f"{rel} still tries to `import d4rl`",
                )


if __name__ == "__main__":
    unittest.main()
