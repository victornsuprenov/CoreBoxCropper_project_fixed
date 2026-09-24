from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .application import ApplicationService
from .config import load_config

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENTS = 1
EXIT_FILE_NOT_FOUND = 2
EXIT_INVALID_IMAGE = 3
EXIT_BOX_NOT_FOUND = 4
EXIT_BOX_NOT_CONFIDENT = 5
EXIT_PROCESSING_ERROR = 6
EXIT_OUTPUT_ERROR = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CoreBoxCropper",
        description="Offline automatic cropper for geological core boxes.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="one source image")
    source.add_argument("--input-dir", help="source directory for batch mode")
    parser.add_argument("--output", help="output image path for one image")
    parser.add_argument("--output-dir", help="output directory for batch mode")
    parser.add_argument("--recursive", action="store_true", help="scan input subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    parser.add_argument("--debug", nargs="?", const=True, help="save intermediate debug images")
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON to stdout")
    parser.add_argument("--no-gui", action="store_true", help="never open the GUI")
    parser.add_argument("--config", help="path to a JSON configuration file")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input and not args.input_dir:
        if args.no_gui:
            return EXIT_INVALID_ARGUMENTS
        from .ui.app import CoreBoxCropperApp
        CoreBoxCropperApp().run()
        return EXIT_SUCCESS
    if args.input and args.output_dir:
        return EXIT_INVALID_ARGUMENTS
    if args.input_dir and (not args.output_dir or args.output):
        return EXIT_INVALID_ARGUMENTS
    debug = args.debug if args.debug is True else Path(args.debug) if args.debug else False
    try:
        config = load_config(Path(args.config) if args.config else None)
        service = ApplicationService(config)
        if args.input:
            result = service.process_image(
                args.input,
                args.output,
                debug=debug,
                overwrite=args.overwrite,
            )
            payload = result.to_dict()
            if args.json:
                print(json.dumps(payload, ensure_ascii=False))
            else:
                print(f"{result.status}: {result.input_path} -> {result.output_path or '-'} ({result.reason})")
            return exit_code_for_result(result)
        results = service.process_directory(
            args.input_dir,
            args.output_dir,
            recursive=args.recursive,
            debug=debug,
            overwrite=args.overwrite,
        )
        payload = {"status": "success" if results and all(item.status == "success" for item in results) else "partial", "results": [item.to_dict() for item in results]}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"processed={len(results)} success={sum(item.status == 'success' for item in results)} error={sum(item.status != 'success' for item in results)}")
        if results and all(item.status == "success" for item in results):
            return EXIT_SUCCESS
        if any(item.reason == "input_directory_not_found" for item in results):
            return EXIT_FILE_NOT_FOUND
        return EXIT_PROCESSING_ERROR
    except (FileNotFoundError, ValueError):
        return EXIT_FILE_NOT_FOUND
    except OSError:
        return EXIT_OUTPUT_ERROR
    except Exception:
        return EXIT_PROCESSING_ERROR


def exit_code_for_result(result) -> int:
    if result.status == "success":
        return EXIT_SUCCESS
    if result.reason in {"file_not_found", "input_directory_not_found"}:
        return EXIT_FILE_NOT_FOUND
    if result.reason in {"invalid_image", "unsupported_image_shape"} or result.reason.startswith("invalid image"):
        return EXIT_INVALID_IMAGE
    if "confidence" in result.reason:
        return EXIT_BOX_NOT_CONFIDENT
    if "not_found" in result.reason:
        return EXIT_BOX_NOT_FOUND
    if "output" in result.reason or "permission" in result.reason:
        return EXIT_OUTPUT_ERROR
    return EXIT_PROCESSING_ERROR


def main() -> int:
    return run_cli(sys.argv[1:])
