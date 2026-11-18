"""Regulariser choices and helpers for building masks in the reconstruction."""

from enum import StrEnum


class RegulariserChoice(StrEnum):
    """Regulariser names accepted in the configuration."""

    DIRECTIONAL_TV = "dTV"
    TV = "TV"
    NONE = "None"


class XrayReference(StrEnum):
    """X-ray reference images for dTV in ``phantoms/xray_refs``, named by their file name part."""

    REALISTIC = "realistic"
    REALISTIC_SLICE = "realistic_slice"
    REALISTIC_SLICE_FILTERED = "realistic_slice_filtered"
    DETERIORATED = "1.0sig_0.02n"


def compute_zero_mask(
    shape: tuple[int, ...],
    radius: int,
    margin: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return the voxel indices lying outside a circular region in each slice.

    Args:
        shape: Shape of the image volume as (slices, height, width).
        radius: Radius of the circular region in pixels.
        margin: Extra margin in pixels added to the radius.

    Returns:
        A tuple of three lists with the slice, row and column indices of the
        voxels outside the region.
    """
    mask = ([], [], [])
    for k in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                # if (i - shape[1] // 2) ** 2 + (j - shape[2] // 2) ** 2 > (radius + margin) ** 2:
                if (i - shape[1] // 2 + 0.5) ** 2 + (j - shape[2] // 2 + 0.5) ** 2 > (radius + margin) ** 2:
                    mask[0].append(k)
                    mask[1].append(i)
                    mask[2].append(j)
    return mask
