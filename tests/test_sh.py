# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""SH block widths and band counts (from cumuli's test_sogst_sh_degree.py)."""

import numpy as np
import pytest

from cumuli_core.sh import (SOGST_SHN_COEFFS, num_sh_bases, rgb_to_sh0,
                            sh0_to_rgb, shn_bands_for_width)


@pytest.mark.parametrize("width, bands", [(9, 1), (24, 2), (45, 3)])
def test_bands_follow_the_block_width(width, bands):
    assert shn_bands_for_width(width) == bands


@pytest.mark.parametrize("width", [0, 15, 30, 47])
def test_other_widths_are_refused(width):
    with pytest.raises(ValueError, match="9, 24 or 45"):
        shn_bands_for_width(width)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_bases_count_includes_dc(degree):
    assert num_sh_bases(degree) == 1 + SOGST_SHN_COEFFS.get(degree, 0)


def test_dc_colour_round_trips():
    rgb = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(sh0_to_rgb(rgb_to_sh0(rgb)), rgb, atol=1e-12)
