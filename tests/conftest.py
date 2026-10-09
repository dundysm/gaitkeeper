import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import synth  # noqa: E402

from gaitkeeper.inject import Harness  # noqa: E402


@pytest.fixture(scope="session")
def truth():
    return synth.make_contract(rounded=False)


@pytest.fixture(scope="session")
def files():
    return synth.make_contract(rounded=True)


@pytest.fixture(scope="session")
def policy():
    return synth.LinearPolicy()


@pytest.fixture(scope="session")
def harness(truth, files, policy):
    g = synth.make_golden()
    first = Harness(g, truth, policy, files).standard()  # fill obs and actions as the policy ran
    g.arrays["obs"], g.arrays["action"] = first["obs"], first["action"]
    return Harness(g, truth, policy, files)


@pytest.fixture(scope="session")
def clean_log(harness):
    return harness.standard()
