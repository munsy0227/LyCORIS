import coverage
import logging
import os
import unittest

from lycoris.logging import logger
from test.functional import LycorisFunctionalTests
from test.lokr import LokrConsistencyTests
from test.module import LycorisModuleTests
from test.wrapper import LycorisWrapperTests
from test.precision_merge_test import MergePrecisionTests
from test.kernels.test_ops import OpsVsFp64
from test.kernels.test_autograd import AutogradParity, SafeFallback
from test.torch_compile import TorchCompileCompatibility

if os.environ.get("LYCORIS_RUN_KOHYA_INTEGRATION") == "1":
    # Importing this module loads a full SDXL checkpoint, so keep it opt-in.
    from test.kohya import LycorisKohyaWrapperTests


cov = coverage.Coverage()
cov.start()

logger.setLevel(logging.ERROR)


TESTS = [
    LycorisModuleTests,
    LycorisFunctionalTests,
    LycorisWrapperTests,
    LokrConsistencyTests,
    MergePrecisionTests,
    OpsVsFp64,
    AutogradParity,
    SafeFallback,
    TorchCompileCompatibility,
]

if os.environ.get("LYCORIS_RUN_KOHYA_INTEGRATION") == "1":
    TESTS.append(LycorisKohyaWrapperTests)


if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=0)
    successful = True
    for test in TESTS:
        suite = test_loader.loadTestsFromTestCase(test)
        result = runner.run(suite)
        successful = result.wasSuccessful() and successful

    cov.stop()
    cov.save()
    cov.report()
    cov.html_report(directory="coverage_report")
    raise SystemExit(0 if successful else 1)
