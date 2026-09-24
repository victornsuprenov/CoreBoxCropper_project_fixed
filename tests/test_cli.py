from coreboxcropper.cli import EXIT_FILE_NOT_FOUND, EXIT_INVALID_ARGUMENTS, build_parser, run_cli


def test_cli_accepts_paths_with_spaces():
    args = build_parser().parse_args(["--input", r"C:\Photos\my box.jpg", "--output", r"C:\Result\out.jpg"])
    assert args.input.endswith("my box.jpg")


def test_cli_requires_batch_output():
    assert run_cli(["--input-dir", "missing", "--no-gui"]) == EXIT_INVALID_ARGUMENTS


def test_cli_debug_without_path_enables_default_directory():
    assert build_parser().parse_args(["--input", "image.jpg", "--debug"]).debug is True


def test_cli_reports_missing_batch_directory():
    assert run_cli(["--input-dir", "missing", "--output-dir", "out", "--no-gui"]) == EXIT_FILE_NOT_FOUND


def test_cli_rejects_mixed_single_and_batch_outputs():
    assert run_cli([
        "--input", "image.jpg", "--output", "out.jpg", "--output-dir", "out", "--no-gui"
    ]) == EXIT_INVALID_ARGUMENTS
