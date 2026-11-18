"""Calculate properties of the geometry of the CT system."""


def get_radius(phantom: str) -> int:
    """Return the phantom radius in mm."""
    match phantom:
        case "gammex":
            return 165
        case "ctp528" | "spiral":
            return 100
        case "toy_inserts" | "toy_water":
            return 8
        case _:
            msg = f"Unknown phantom: {phantom}"
            raise ValueError(msg)


def get_detector_distance(phantom: str, dataset: str) -> int:
    """Return the phantom-to-detector distance in mm, for the ``ideal`` or ``realistic`` dataset."""
    match phantom, dataset:
        case "toy_inserts" | "toy_water", "ideal":
            return 2
        case _, "ideal":
            return 10
        case _, "realistic":
            return 150
        case _:
            msg = f"Unknown phantom or dataset: {phantom}, {dataset}"
            raise ValueError(msg)


def get_centre_of_rotation(phantom: str, dataset: str) -> float:
    """Return the position of the centre of rotation: phantom radius plus detector distance."""
    return get_radius(phantom) + get_detector_distance(phantom, dataset)


def get_image_halfspace_xz(phantom: str) -> float:
    """Return the half-width of the image in the xz plane, the same for both datasets."""
    return get_radius(phantom) + get_detector_distance(phantom, "ideal")


def get_image_w(phantom: str, pixel_w: float) -> int:
    """Return the image width in pixels for a pixel width in mm."""
    return int(2 * get_image_halfspace_xz(phantom) / pixel_w)


def get_image_halfspace_yz(n_slices: int, slice_thickness: float) -> float:
    """Return the half-height of the image in the yz plane."""
    return n_slices * slice_thickness / 2
