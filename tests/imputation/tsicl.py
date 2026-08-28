"""
Test cases for TSICL imputation model.
"""

# Created by Nicolas Vautier <nicolas.vautier@edf.fr>
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest

from pypots.imputation import TSICL
from pypots.nn.functional import calc_mse
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
    DEVICE,
    TEST_SET,
    GENERAL_H5_TEST_SET_PATH,
)


class TestTSICL(unittest.TestCase):
    logger.info("Running tests for an imputation model TSICL...")

    tsicl = TSICL(device=DEVICE)

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_0_impute(self):
        with pytest.warns(UserWarning):
            self.tsicl.fit(TEST_SET)

        imputation_results = self.tsicl.predict(TEST_SET)
        assert not np.isnan(imputation_results["imputation"]).any(), (
            "Output still has missing values after running impute()."
        )

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"TSICL test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_1_lazy_loading(self):
        imputation_results = self.tsicl.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(imputation_results["imputation"]).any(), (
            "Output still has missing values after running impute()."
        )

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading TSICL test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
