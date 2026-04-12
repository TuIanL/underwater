from __future__ import annotations

import argparse


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        return
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swim-pose")
    subparsers = parser.add_subparsers(dest="command")

    manifest_parser = subparsers.add_parser("manifest", help="Create and audit clip manifests.")
    manifest_sub = manifest_parser.add_subparsers(dest="manifest_command")

    manifest_init = manifest_sub.add_parser("init", help="Initialize a clip manifest from a video root.")
    manifest_init.add_argument("--video-root", required=True)
    manifest_init.add_argument("--output", required=True)
    manifest_init.set_defaults(handler=_handle_manifest_init)

    manifest_audit = manifest_sub.add_parser("audit-sync", help="Audit dual-view synchronization metadata.")
    manifest_audit.add_argument("--manifest", required=True)
    manifest_audit.add_argument("--output", required=True)
    manifest_audit.set_defaults(handler=_handle_manifest_audit)

    frames_parser = subparsers.add_parser("frames", help="Extract frame images from raw videos.")
    frames_sub = frames_parser.add_subparsers(dest="frames_command")

    frames_extract = frames_sub.add_parser("extract", help="Extract sampled frames from manifest videos.")
    frames_extract.add_argument("--manifest", required=True)
    frames_extract.add_argument("--output-root", required=True)
    frames_extract.add_argument("--index-output", required=True)
    frames_extract.add_argument("--views", nargs="+", default=["above", "under"])
    frames_extract.add_argument("--every-nth", type=int, default=1)
    frames_extract.set_defaults(handler=_handle_frames_extract)

    annotation_parser = subparsers.add_parser("annotations", help="Work with annotation templates and validation.")
    annotation_sub = annotation_parser.add_subparsers(dest="annotation_command")

    annotations_template = annotation_sub.add_parser("template", help="Write a blank frame annotation template.")
    annotations_template.add_argument("--output", required=True)
    annotations_template.set_defaults(handler=_handle_annotation_template)

    annotations_validate = annotation_sub.add_parser("validate", help="Validate a frame annotation JSON file.")
    annotations_validate.add_argument("--input", required=True)
    annotations_validate.set_defaults(handler=_handle_annotation_validate)

    annotations_index = annotation_sub.add_parser("index", help="Index validated annotation files into CSV.")
    annotations_index.add_argument("--annotation-root", required=True)
    annotations_index.add_argument("--output", required=True)
    annotations_index.set_defaults(handler=_handle_annotation_index)

    annotations_scaffold = annotation_sub.add_parser("scaffold", help="Create blank annotation JSON files from selected seed frames.")
    annotations_scaffold.add_argument("--seed-csv", required=True)
    annotations_scaffold.add_argument("--frame-root", required=True)
    annotations_scaffold.add_argument("--output-root", required=True)
    annotations_scaffold.set_defaults(handler=_handle_annotation_scaffold)

    annotations_gui = annotation_sub.add_parser("gui", help="Open a local GUI for keypoint annotation and review.")
    annotations_gui.add_argument("--annotation-root", required=True)
    annotations_gui.add_argument("--frame-root", required=True)
    annotations_gui.set_defaults(handler=_handle_annotation_gui)

    annotations_web = annotation_sub.add_parser("web", help="Open a browser-based annotation UI.")
    annotations_web.add_argument("--annotation-root", required=True)
    annotations_web.add_argument("--frame-root", required=True)
    annotations_web.add_argument("--host", default="127.0.0.1")
    annotations_web.add_argument("--port", type=int, default=8765)
    annotations_web.add_argument("--no-browser", action="store_true")
    annotations_web.set_defaults(handler=_handle_annotation_web)

    annotations_audit = annotation_sub.add_parser("audit", help="Audit labeled annotations for common consistency issues.")
    annotations_audit.add_argument("--annotation-root", required=True)
    annotations_audit.add_argument("--output", required=True)
    annotations_audit.set_defaults(handler=_handle_annotation_audit)

    seed_parser = subparsers.add_parser("seed", help="Seed frame selection tooling.")
    seed_sub = seed_parser.add_subparsers(dest="seed_command")

    seed_select = seed_sub.add_parser("select", help="Select diverse seed frames from a manifest.")
    seed_select.add_argument("--manifest", required=True)
    seed_select.add_argument("--output", required=True)
    seed_select.add_argument("--frames-per-clip", type=int, default=12)
    seed_select.add_argument("--source-view", default="stitched", choices=["above", "under", "stitched"])
    seed_select.add_argument("--seed", type=int, default=7)
    seed_select.set_defaults(handler=_handle_seed_select)

    dataset_parser = subparsers.add_parser("dataset", help="Dataset split utilities.")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command")

    dataset_split = dataset_sub.add_parser("split", help="Create train/val/test splits grouped by athlete or session.")
    dataset_split.add_argument("--index", required=True)
    dataset_split.add_argument("--output-dir", required=True)
    dataset_split.add_argument("--group-by", nargs="+", default=["athlete_id", "session_id"])
    dataset_split.add_argument("--val-ratio", type=float, default=0.15)
    dataset_split.add_argument("--test-ratio", type=float, default=0.15)
    dataset_split.add_argument("--seed", type=int, default=7)
    dataset_split.set_defaults(handler=_handle_dataset_split)

    train_parser = subparsers.add_parser("train", help="Training entrypoints.")
    train_sub = train_parser.add_subparsers(dest="train_command")

    train_sup = train_sub.add_parser("supervised", help="Run supervised training.")
    train_sup.add_argument("--config", required=True)
    train_sup.set_defaults(handler=_handle_train_supervised)

    train_semi = train_sub.add_parser("semisupervised", help="Run semi-supervised training.")
    train_semi.add_argument("--config", required=True)
    train_semi.set_defaults(handler=_handle_train_semisupervised)

    predict_parser = subparsers.add_parser("predict", help="Run model inference and export keypoint predictions.")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--index", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--unlabeled", action="store_true")
    predict_parser.set_defaults(handler=_handle_predict)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against annotations.")
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--annotations", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.set_defaults(handler=_handle_evaluate)

    pseudo_parser = subparsers.add_parser("pseudolabel", help="Pseudo-label tooling.")
    pseudo_sub = pseudo_parser.add_subparsers(dest="pseudo_command")

    pseudo_generate = pseudo_sub.add_parser("generate", help="Generate pseudo-labels from model predictions.")
    pseudo_generate.add_argument("--predictions", required=True)
    pseudo_generate.add_argument("--output", required=True)
    pseudo_generate.add_argument("--threshold", type=float, default=0.5)
    pseudo_generate.set_defaults(handler=_handle_pseudolabel_generate)

    return parser


def _handle_manifest_init(args: argparse.Namespace) -> None:
    from .manifest import discover_manifest, write_manifest

    entries = discover_manifest(args.video_root)
    write_manifest(args.output, entries)
    print(f"Wrote {len(entries)} manifest rows to {args.output}")


def _handle_manifest_audit(args: argparse.Namespace) -> None:
    from .io import write_csv_rows
    from .manifest import MANIFEST_FIELDS, audit_manifest

    rows = audit_manifest(args.manifest)
    write_csv_rows(args.output, MANIFEST_FIELDS, rows)
    print(f"Wrote audited manifest with {len(rows)} rows to {args.output}")


def _handle_frames_extract(args: argparse.Namespace) -> None:
    from .frames import extract_frames_from_manifest

    destination = extract_frames_from_manifest(
        manifest_path=args.manifest,
        output_root=args.output_root,
        index_output=args.index_output,
        views=tuple(args.views),
        every_nth=args.every_nth,
    )
    print(f"Wrote extracted frame index to {destination}")


def _handle_annotation_template(args: argparse.Namespace) -> None:
    from .annotations import write_template

    destination = write_template(args.output)
    print(f"Wrote annotation template to {destination}")


def _handle_annotation_validate(args: argparse.Namespace) -> None:
    from .annotations import validate_file

    errors = validate_file(args.input)
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))
    print(f"{args.input} is valid")


def _handle_annotation_index(args: argparse.Namespace) -> None:
    from .annotations import build_annotation_index

    destination = build_annotation_index(args.annotation_root, args.output)
    print(f"Wrote annotation index to {destination}")


def _handle_annotation_scaffold(args: argparse.Namespace) -> None:
    from .annotations import scaffold_annotations

    created = scaffold_annotations(args.seed_csv, args.frame_root, args.output_root)
    print(f"Created {len(created)} scaffold annotation files in {args.output_root}")


def _handle_annotation_gui(args: argparse.Namespace) -> None:
    from .annotation_gui import AnnotationGui

    app = AnnotationGui(args.annotation_root, args.frame_root)
    app.run()


def _handle_annotation_web(args: argparse.Namespace) -> None:
    from .annotation_web import run_annotation_web

    run_annotation_web(
        annotation_root=args.annotation_root,
        frame_root=args.frame_root,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def _handle_annotation_audit(args: argparse.Namespace) -> None:
    from .audit import audit_annotations

    destination = audit_annotations(args.annotation_root, args.output)
    print(f"Wrote annotation audit to {destination}")


def _handle_seed_select(args: argparse.Namespace) -> None:
    from .sampling import select_seed_frames

    destination = select_seed_frames(
        manifest_path=args.manifest,
        output_path=args.output,
        frames_per_clip=args.frames_per_clip,
        source_view=args.source_view,
        seed=args.seed,
    )
    print(f"Wrote seed frame selection to {destination}")


def _handle_dataset_split(args: argparse.Namespace) -> None:
    from .sampling import create_group_splits

    destinations = create_group_splits(
        index_path=args.index,
        output_dir=args.output_dir,
        group_by=tuple(args.group_by),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for destination in destinations:
        print(f"Wrote {destination}")


def _handle_train_supervised(args: argparse.Namespace) -> None:
    from .training.supervised import run_supervised_training

    checkpoint = run_supervised_training(args.config)
    print(f"Saved supervised checkpoint to {checkpoint}")


def _handle_train_semisupervised(args: argparse.Namespace) -> None:
    from .training.semisupervised import run_semi_supervised_training

    checkpoint = run_semi_supervised_training(args.config)
    print(f"Saved semi-supervised checkpoint to {checkpoint}")


def _handle_predict(args: argparse.Namespace) -> None:
    from .training.inference import run_inference

    destination = run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        index_path=args.index,
        output_path=args.output,
        labeled=not args.unlabeled,
    )
    print(f"Wrote predictions to {destination}")


def _handle_evaluate(args: argparse.Namespace) -> None:
    from .training.evaluate import evaluate_predictions_file

    destination = evaluate_predictions_file(args.predictions, args.annotations, args.output)
    print(f"Wrote evaluation report to {destination}")


def _handle_pseudolabel_generate(args: argparse.Namespace) -> None:
    from .training.pseudolabels import generate_pseudolabel_file

    destination = generate_pseudolabel_file(args.predictions, args.output, args.threshold)
    print(f"Wrote pseudolabels to {destination}")


if __name__ == "__main__":
    main()
