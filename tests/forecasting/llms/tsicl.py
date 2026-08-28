"""
Test cases for TSICL forecasting model.
"""

# Created by Nicolas Vautier <nicolas.vautier@edf.fr>
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest

from pypots.forecasting import TSICL
from pypots.nn.functional import calc_mse
from pypots.utils.logging import logger
from tests.global_test_config import (
    DEVICE,
    N_PRED_STEPS,
    FORECASTING_TEST_SET,
    FORECASTING_H5_TEST_SET_PATH,
)


class TestTSICL(unittest.TestCase):
    logger.info("Running tests for a forecasting model TSICL...")

    tsicl = TSICL(n_pred_steps=N_PRED_STEPS, device=DEVICE)

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_0_forecasting(self):
        with pytest.warns(UserWarning):
            self.tsicl.fit(FORECASTING_TEST_SET)

        forecasting_X = self.tsicl.predict(FORECASTING_TEST_SET)["forecasting"]
        assert not np.isnan(forecasting_X).any(), (
            "Output has missing values in the forecasting results that should not be."
        )
        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"TSICL test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_1_lazy_loading(self):
        forecasting_results = self.tsicl.predict(FORECASTING_H5_TEST_SET_PATH)
        forecasting_X = forecasting_results["forecasting"]
        assert not np.isnan(forecasting_X).any(), (
            "Output has missing values in the forecasting results that should not be."
        )

        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"Lazy-loading TSICL test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
