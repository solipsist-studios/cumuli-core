# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Spherical-harmonic block layout shared by the PLY, the packer and the renderer.

An f_rest block is [N, 3 * coeffs], channel-major: column j * coeffs + k is
channel j, coefficient k. coeffs is (bands + 1)^2 - 1, so 3, 8 or 15 for 1, 2
or 3 bands. Band count 0 is a model with no f_rest block at all.

Numpy free, torch free.
"""

# Higher-order SH coefficients per colour channel for each band count.
SOGST_SHN_COEFFS = {1: 3, 2: 8, 3: 15}

# shN centroid texture widths a .sogst decoder accepts. The decoder infers
# the band count from this width, so it is normative (docs/sogst-format.md
# section 4.7).
SOGST_SHN_WIDTHS = {1: 192, 2: 512, 3: 960}

# Zeroth-order SH basis constant: rgb = 0.5 + SH_C0 * f_dc.
SH_C0 = 0.28209479177387814


def shn_bands_for_width(width):
    """The SH band count of an f_rest block `width` columns wide (9, 24 or
    45).  Raises on any other width: a partial block would pack garbage into
    the SH textures, and nothing downstream would notice."""
    for bands, coeffs in SOGST_SHN_COEFFS.items():
        if width == 3 * coeffs:
            return bands
    raise ValueError(f'f_rest has {width} columns; it must be 3 * coeffs for '
                     f'1, 2 or 3 SH bands (9, 24 or 45)')


def num_sh_bases(degree):
    """Total SH coefficients per channel at `degree`, DC included: (d+1)^2."""
    if degree not in (0, 1, 2, 3):
        raise ValueError(f'SH degree must be 0, 1, 2 or 3, got {degree}')
    return (degree + 1) ** 2


def rgb_to_sh0(rgb):
    """Map colour in [0, 1] to the DC coefficient that renders it."""
    return (rgb - 0.5) / SH_C0


def sh0_to_rgb(f_dc):
    """Inverse of rgb_to_sh0."""
    return f_dc * SH_C0 + 0.5
