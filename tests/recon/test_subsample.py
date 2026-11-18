"""Tests of ``recon/utils/subsample_matrix_vector_pair.py`` with the pct-tools call replaced."""

from recon.utils import subsample_matrix_vector_pair as module


def test_subsample_passes_each_chunk_of_the_requested_projections(tmp_path, mocker):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    matrix_dir = input_dir / "matrix-0.5mm"
    matrix_dir.mkdir(parents=True)

    for name in ["0.0-0.mat", "0.0-1.mat", "180.0.mat", "90.0.mat", "0.0-0.vec"]:
        (matrix_dir / name).touch()

    pct_tools_call = mocker.patch.object(module, "subsample_matrix_vector_pair")

    module.subsample(input_dir, output_dir, n_projections=2, pixel_width=0.5, ratio=0.2)

    calls = pct_tools_call.call_args_list
    # Two projections (0 and 180 degrees): 90.0 is skipped and so is the .vec file.
    assert sorted(args[1] for args, _ in calls) == ["0.0-0", "0.0-1", "180.0"]
    assert all(args[0] == str(input_dir) and args[2] == str(output_dir) for args, _ in calls)
    assert all(args[3:] == (0.2, 0.5) for args, _ in calls)
    assert all(kwargs == {"compression_level": 9} for _, kwargs in calls)
    assert (output_dir / "matrix-0.5mm").is_dir()
    assert (output_dir / "b").is_dir()
