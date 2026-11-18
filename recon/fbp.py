"""Bin proton WEPL into sinograms and reconstruct FBP images for a phantom."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import odl
import pandas as pd
from odl.applications import tomo
from scipy.ndimage import median_filter
from scipy.stats import binned_statistic_2d

Y_BIN_SIZE = 1.0  # slice thickness [mm]
Y_HALFSPACE = 1.5  # half the extent of the three slices [mm]
MARGIN = 10  # distance from the phantom to the edge of the image [mm]


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phantom", type=str, required=True, help="Setup, of the form phantom-dataset")
    parser.add_argument("--width", type=int, required=True, help="Image width in pixels")
    parser.add_argument("--pixel-size", type=float, required=True, help="Pixel size in mm")
    parser.add_argument(
        "--protons-dir",
        type=Path,
        required=True,
        help="Directory with the processed proton dataframes (df_<angle>.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory the sinogram and the images are written to (FBP_DIR of the reconstruction)",
    )
    parser.add_argument("--dose", type=float, default=5, help="Target dose in mGy")
    parser.add_argument("--actual-dose", type=float, default=5, help="Dose of the proton dataframes in mGy")
    parser.add_argument("-n", type=int, default=720, help="Number of projections")
    return parser.parse_args()


def bin_quantity(
    x: np.ndarray,
    y: np.ndarray,
    val: np.ndarray,
    xbins: np.ndarray,
    ybins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin the mean, standard deviation and count of a quantity on a 2D grid.

    Args:
        x: The x-coordinates of the samples.
        y: The y-coordinates of the samples.
        val: The values to bin.
        xbins: The bin edges along x.
        ybins: The bin edges along y.

    Returns:
        The mean per bin with empty bins set to zero, the standard deviation per bin, and the
        number of samples per bin.
    """
    mean = binned_statistic_2d(x, y, val, statistic="mean", bins=[xbins, ybins])[0]
    count = binned_statistic_2d(x, y, val, statistic="count", bins=[xbins, ybins])[0]
    mean = np.nan_to_num(mean, nan=0.0)
    std = binned_statistic_2d(x, y, val, statistic="std", bins=[xbins, ybins])[0]

    return mean, std, count


def bin_wepl(protons: pd.DataFrame, xbins: np.ndarray, ybins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin the WEPL of the protons by their position on the midplane.

    Args:
        protons: Protons with their midplane positions ``x_mid``, ``y_mid`` and their ``wepl``.
        xbins: The bin edges along x.
        ybins: The bin edges along y.

    Returns:
        The mean WEPL per bin, with empty bins set to zero, and the number of protons per bin.
    """
    x = protons["x_mid"].to_numpy()
    y = protons["y_mid"].to_numpy()
    wepl = protons["wepl"].to_numpy()

    mean_wepl, _, count = bin_quantity(x, y, wepl, xbins, ybins)
    return mean_wepl, count


def format_dose(dose: float) -> str:
    """Format a dose for a file name, as ``recon/algo/fista.py`` does: 5.0 gives "5", 0.2 gives "0.2"."""
    return str(int(dose)) if float(dose).is_integer() else str(dose)


def compute_sinogram(
    protons_dir: Path,
    n_projections: int,
    dose_ratio: float,
    xbins: np.ndarray,
    ybins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin the mean WEPL of every projection into a sinogram.

    Args:
        protons_dir: Directory with one ``df_<angle>.pkl`` per projection.
        n_projections: Number of projections over a full rotation.
        dose_ratio: Fraction of the protons of each projection to use, drawn at random.
        xbins: The detector bin edges along x.
        ybins: The slice edges along y.

    Returns:
        The mean WEPL and the number of protons, each of shape (projections, x bins, slices).
    """
    sinogram = np.zeros((n_projections, len(xbins) - 1, len(ybins) - 1))
    count = np.zeros((n_projections, len(xbins) - 1, len(ybins) - 1))

    increment = 360 / n_projections

    for proj in np.arange(0, 360, increment):
        print(f"proj {proj}")

        with Path(protons_dir, f"df_{proj}.pkl").open("rb") as f:
            protons = pickle.load(f)  # noqa: S301

        protons = protons.sample(n=int(len(protons) * dose_ratio), random_state=0)
        proj_mean_wepl, count_proj = bin_wepl(protons, xbins, ybins)

        sinogram[int(proj / increment)] = proj_mean_wepl
        count[int(proj / increment)] = count_proj

    return sinogram, count


def reconstruct(sino: np.ndarray, im_halfspace: float, im_shape: tuple[int, int]) -> np.ndarray:
    """Reconstruct one slice of the sinogram by filtered back-projection with ODL.

    Args:
        sino: Sinogram of one slice, of shape (projections, x bins), over a full rotation.
        im_halfspace: Half the width of the image and of the detector in mm.
        im_shape: Shape of the image in pixels.

    Returns:
        The reconstructed image.
    """
    n_angles = sino.shape[0]
    det_shape = sino.shape[1]

    image_space = odl.uniform_discr(
        min_pt=[-im_halfspace, -im_halfspace],
        max_pt=[im_halfspace, im_halfspace],
        shape=im_shape,
        dtype=np.float32,
    )

    angle_partition = odl.uniform_partition(0, 2 * np.pi, n_angles)
    detector_partition = odl.uniform_partition(-im_halfspace, im_halfspace, det_shape)
    geometry = tomo.Parallel2dGeometry(angle_partition, detector_partition)

    ray_trafo = tomo.RayTransform(image_space, geometry, impl="skimage")

    fbp = tomo.fbp_op(ray_trafo, filter_type="Shepp-Logan", frequency_scaling=1)
    proj_data = ray_trafo.range.element(sino)

    fbp_recon = fbp(proj_data)
    im = fbp_recon.asarray().copy()
    return np.flipud(im)


def mask_outside_phantom(im: np.ndarray, im_radius: int, margin: int = 1) -> np.ndarray:
    """Set the pixels further than ``im_radius + margin`` pixels from the image centre to zero.

    Args:
        im: The image, modified in place.
        im_radius: Radius of the phantom in pixels.
        margin: Number of pixels kept beyond the radius.

    Returns:
        The masked image.
    """
    c = im.shape[0] // 2
    i, j = np.ogrid[: im.shape[0], : im.shape[1]]
    im[(i - c + 0.5) ** 2 + (j - c + 0.5) ** 2 > (im_radius + margin) ** 2] = 0
    return im


def main() -> None:
    """Compute the sinogram of a setup and write its FBP reconstruction."""
    args = parse_args()
    phantom_str = args.phantom
    image_width = args.width
    pixel_size = args.pixel_size
    n_projections = args.n
    target_dose = format_dose(args.dose)
    outdir: Path = args.output_dir

    phantom = phantom_str.split("-")[0]

    print(
        f"Computing sinogram and reconstruct {phantom_str} at {target_dose} mGy "
        f"with {n_projections} projections",
    )

    outdir.mkdir(parents=True, exist_ok=True)

    xbinsize = pixel_size * 2

    radius = 165 if phantom == "gammex" else 100  # phantom radius [mm]
    im_radius = int(radius * 1.0 / pixel_size)
    im_halfspace = radius + MARGIN

    xbins = np.arange(-im_halfspace, im_halfspace + xbinsize, xbinsize)
    ybins = np.arange(-Y_HALFSPACE, Y_HALFSPACE + Y_BIN_SIZE, Y_BIN_SIZE)

    sinogram, count = compute_sinogram(
        args.protons_dir,
        n_projections,
        args.dose / args.actual_dose,
        xbins,
        ybins,
    )

    np.save(outdir / f"sino_{target_dose}mGy_{image_width}pix.npy", sinogram)
    np.save(outdir / f"count_{target_dose}mGy_{image_width}pix.npy", count)

    im = reconstruct(sinogram[..., 0], im_halfspace, (image_width, image_width))
    np.save(outdir / f"fbp_raw_{target_dose}mGy_{image_width}pix.npy", im)

    im = mask_outside_phantom(im, im_radius)
    im[im < 0] = 0

    im = median_filter(im, size=3)
    np.save(outdir / f"fbp_{target_dose}mGy_{image_width}pix.npy", im)


if __name__ == "__main__":
    main()
