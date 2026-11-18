"""Helper functions for generating phantom images and simulated x-ray reconstructions."""

import numpy as np
from loguru import logger
from skimage.draw import disk
from skimage.filters import gaussian
from skimage.transform import iradon, radon

from phantoms.constants import MU_AIR


def deteriorate_image(img: np.ndarray, sigma: float, noise_level: float, random_seed: int = 1) -> np.ndarray:
    """Deteriorate an image by blurring and adding noise.

    Args:
        img: The image to deteriorate.
        sigma: The standard deviation of the Gaussian blur.
        noise_level: The standard deviation of the Gaussian noise expressed as a fraction.
        random_seed: Seed of the noise generator.

    Returns:
        The deteriorated image.
    """
    rng = np.random.default_rng(random_seed)

    img = gaussian(img, sigma=sigma)
    noise = noise_level * np.max(img) * rng.standard_normal(img.shape)[img > MU_AIR]
    noise[noise < 0] = 0
    img[img > MU_AIR] += noise
    return img


def get_circle_image(
    radius: float,
    pixel_width: float,
    background_value: float,
    circle_value: float,
    circle_to_edge_distance: float = 2,
) -> np.ndarray:
    """Draw a circle of a given radius in a 2D array of a given pixel width."""
    radius_in_pixels = int(radius / pixel_width)
    width_in_pixels = int(2 * (radius + circle_to_edge_distance) / pixel_width)

    im_shape = (width_in_pixels, width_in_pixels)
    centre = (width_in_pixels // 2 - 0.5, width_in_pixels // 2 - 0.5)
    # centre = (width_in_pixels // 2, width_in_pixels // 2)

    im = np.zeros(im_shape)
    im[:] = background_value
    im[disk(centre, radius_in_pixels, shape=im_shape)] = circle_value

    return im


def draw_circle_in_image(
    im: np.ndarray,
    centre_in_mm: tuple[float, float],
    radius: float,
    pixel_width: float,
    circle_value: float,
) -> np.ndarray:
    """Draw a circle of a given radius in a 2D array of a given pixel width."""
    radius_in_pixels = int(radius / pixel_width)

    # centre = (centre_in_mm[0] / pixel_width, centre_in_mm[1] / pixel_width)
    centre = (centre_in_mm[0] / pixel_width - 0.5, centre_in_mm[1] / pixel_width - 0.5)
    im[disk(centre, radius_in_pixels)] = circle_value

    return im


def apply_pre_log_noise(
    sinogram: np.ndarray,
    initial_intensity: float = 5e5,
    pixel_w: float = 1.0,
    t_exposure: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply Poisson noise to the detected counts of a sinogram, before the logarithm.

    Args:
        sinogram: The sinogram data to which noise will be applied.
        initial_intensity: The initial intensity of the signal before noise is applied. A conservative
            estimate is 1e8 photons/mm^2 per second which with an assumed exposure of 5 ms corresponds to
            5e5 counts/mm^2.
        pixel_w: The width of the pixel in mm. This is used to scale the initial intensity to match the
            required pixel area.
        t_exposure: The total exposure time per view in seconds. This is used to scale the initial intensity.
        rng: Random number generator for the noise; a fresh unseeded one if not given.

    Returns:
        The noisy sinogram, with counts clipped to at least one before taking the logarithm.
    """
    if rng is None:
        rng = np.random.default_rng()

    initial_intensity = initial_intensity * pixel_w**2 * t_exposure / sinogram.shape[1]

    logger.info(f"Number of views: {sinogram.shape[1]}")
    logger.info(f"Initial intensity per pixel: {initial_intensity:.3e} counts")

    counts = rng.poisson(initial_intensity * np.exp(-sinogram))
    counts = np.clip(counts, 1, None)  # avoid zeros
    return -np.log(counts / initial_intensity)


def generate_realistic_ct_image(
    im: np.ndarray,
    theta: np.ndarray,
    image_w: int = 220,
    pixel_w: float = 1.0,
    initial_intensity: float = 5e5,
    t_exposure: float = 0.3,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a noisy x-ray CT reconstruction of an attenuation image.

    The image is converted from 1/cm to 1/mm and forward projected; pre-log noise is
    added to the sinogram, which is reconstructed by filtered backprojection and
    converted back to 1/cm.

    Args:
        im: The input image to be processed, with attenuation values in 1/cm.
        theta: The angles at which the projections are taken.
        image_w: The width of the output image in pixels.
        pixel_w: The width of each pixel in mm.
        initial_intensity: The initial intensity of the signal before noise is applied.
        t_exposure: The total exposure time per view in seconds.
        random_seed: Seed of the noise generator shared by all slices; unseeded if not given.

    Returns:
        The mean of the 12 noisy reconstructions, which mimics a thick slice, and the last single
        noisy reconstruction.
    """
    rng = np.random.default_rng(random_seed)

    n_pixel_per_cm = 1 / pixel_w * 10
    im_scaled = im / n_pixel_per_cm  # scale attenuation from 1/cm to 1/mm

    sinogram = radon(im_scaled, theta=theta, circle=False)

    n_samples = 12

    slices = []
    for i in range(n_samples):
        logger.info(f"slice {i + 1}/{n_samples}")
        sinogram_noisy = apply_pre_log_noise(
            sinogram,
            initial_intensity=initial_intensity,
            pixel_w=pixel_w,
            t_exposure=t_exposure,
            rng=rng,
        )

        recon_noisy = iradon(sinogram_noisy, theta=theta, filter_name="cosine", output_size=image_w)
        recon_noisy *= n_pixel_per_cm  # scale back to 1/cm

        slices.append(recon_noisy)

    recon_noisy_thick_slice = np.mean(slices, axis=0)

    return recon_noisy_thick_slice, recon_noisy
