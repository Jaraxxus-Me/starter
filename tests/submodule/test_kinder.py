"""Tests for the kindergarden (kinder) submodule."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add kindergarden's src directory to sys.path so we can import kinder
# without installing the full package (which has heavy simulation deps).
_KINDER_SRC = (
    Path(__file__).resolve().parent.parent.parent
    / "third-party"
    / "kindergarden"
    / "src"
)
sys.path.insert(0, str(_KINDER_SRC))


def test_submodule_exists():
    """The kindergarden submodule directory and key files are present."""
    kinder_root = _KINDER_SRC.parent
    assert kinder_root.is_dir()
    assert (kinder_root / "pyproject.toml").is_file()
    assert (_KINDER_SRC / "kinder" / "__init__.py").is_file()
    assert (_KINDER_SRC / "kinder" / "core.py").is_file()
    assert (_KINDER_SRC / "kinder" / "utils.py").is_file()
    assert (_KINDER_SRC / "kinder" / "wrappers.py").is_file()


def test_env_categories_exist():
    """The expected environment category directories exist."""
    envs_dir = _KINDER_SRC / "kinder" / "envs"
    assert envs_dir.is_dir()
    for category in ["kinematic2d", "dynamic2d", "kinematic3d", "dynamic3d"]:
        assert (envs_dir / category).is_dir(), f"Missing env category: {category}"


def test_import_kinder():
    """The kinder package can be imported (top-level, no heavy deps)."""
    import kinder  # pylint: disable=import-outside-toplevel

    assert hasattr(kinder, "register_all_environments")
    assert hasattr(kinder, "get_all_env_ids")
    assert hasattr(kinder, "get_env_classes")
    assert hasattr(kinder, "get_env_variants")
    assert hasattr(kinder, "get_env_categories")
    assert hasattr(kinder, "make")
    assert hasattr(kinder, "ENV_CLASSES")


def test_check_deps():
    """_check_deps returns True for stdlib, False for missing packages."""
    from kinder import (  # pylint: disable=import-outside-toplevel
        _check_deps,
    )

    assert _check_deps("os", "sys")
    assert not _check_deps("nonexistent_package_xyz_12345")
    assert not _check_deps("os", "nonexistent_package_xyz_12345")


def test_register_all_environments_no_crash():
    """register_all_environments runs without error even if deps are missing."""
    import kinder  # pylint: disable=import-outside-toplevel

    kinder.register_all_environments()
    # Should return a set (possibly empty if no category deps are installed).
    env_ids = kinder.get_all_env_ids()
    assert isinstance(env_ids, set)


def test_get_env_classes_returns_dict():
    """get_env_classes returns a dict mapping class names to metadata."""
    import kinder  # pylint: disable=import-outside-toplevel

    kinder.register_all_environments()
    classes = kinder.get_env_classes()
    assert isinstance(classes, dict)
    for name, meta in classes.items():
        assert isinstance(name, str)
        assert "entry_point" in meta
        assert "category" in meta
        assert "variants" in meta


def test_get_env_categories_returns_dict():
    """get_env_categories returns categories mapping to class name lists."""
    import kinder  # pylint: disable=import-outside-toplevel

    kinder.register_all_environments()
    categories = kinder.get_env_categories()
    assert isinstance(categories, dict)
    for cat, class_names in categories.items():
        assert isinstance(cat, str)
        assert isinstance(class_names, list)


def test_import_wrappers():
    """NoisyObservation and NoisyAction wrappers are importable."""
    from kinder.wrappers import (  # pylint: disable=import-outside-toplevel
        NoisyAction,
        NoisyObservation,
    )

    assert callable(NoisyAction)
    assert callable(NoisyObservation)


def test_noisy_observation_on_mock_env():
    """NoisyObservation adds noise to a simple mock environment."""
    import gymnasium  # pylint: disable=import-outside-toplevel
    from kinder.wrappers import (  # pylint: disable=import-outside-toplevel
        NoisyObservation,
    )

    # Use a simple built-in gym env with Box observation space.
    env = gymnasium.make("MountainCar-v0")
    wrapped = NoisyObservation(env, noise_std=0.1)
    obs, _ = wrapped.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    wrapped.close()


def test_noisy_action_on_mock_env():
    """NoisyAction adds noise to a simple mock environment."""
    import gymnasium  # pylint: disable=import-outside-toplevel
    from kinder.wrappers import (  # pylint: disable=import-outside-toplevel
        NoisyAction,
    )

    # Use a built-in gym env with Box action space.
    env = gymnasium.make("MountainCarContinuous-v0")
    wrapped = NoisyAction(env, noise_std=0.1)
    wrapped.reset(seed=42)
    action = wrapped.action_space.sample()
    obs, _, _, _, _ = wrapped.step(action)
    assert isinstance(obs, np.ndarray)
    wrapped.close()


def test_get_env_variants_keyerror():
    """get_env_variants raises KeyError for unregistered class names."""
    import kinder  # pylint: disable=import-outside-toplevel

    with pytest.raises(KeyError):
        kinder.get_env_variants("NonexistentEnvClass12345")
