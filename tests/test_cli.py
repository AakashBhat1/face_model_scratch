from pathlib import Path

import pytest

from face_model_core.cli import _build_parser


def test_cli_parsing_train_command() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train", "--data-root", str(Path("data"))])
    assert args.command == "train"
    assert args.backbone == "resnet50"
    assert args.mixed_precision is True


def test_cli_train_allows_disabling_mixed_precision() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train", "--data-root", str(Path("data")), "--no-mixed-precision"])
    assert args.mixed_precision is False


def test_cli_parsing_infer_command() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--image",
            "a.jpg",
            "--checkpoint",
            "b.pt",
            "--threshold",
            "0.5",
        ]
    )
    assert args.command == "infer"
    assert abs(args.threshold - 0.5) < 1e-9


def test_cli_rejects_out_of_range_threshold() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "infer",
                "--image",
                "a.jpg",
                "--checkpoint",
                "b.pt",
                "--threshold",
                "1.5",
            ]
        )


def test_cli_rejects_non_finite_threshold() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "infer",
                "--image",
                "a.jpg",
                "--checkpoint",
                "b.pt",
                "--threshold",
                "nan",
            ]
        )
