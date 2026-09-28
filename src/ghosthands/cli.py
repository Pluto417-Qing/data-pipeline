from __future__ import annotations

import json
from pathlib import Path

import typer

from .backends import create_backend
from .branches import BRANCHES, add_branch_metadata, resolve_branch
from .direct import DEFAULT_GHOST_HAND_PROMPT, candidate_manifest, resolve_direct_engine, run_dashscope_video_edit, run_ltx_video_edit, run_runway_video_edit
from .pipeline import find_videos, process_video
from .report import build_report

app = typer.Typer(add_completion=False, help="Create and review hand-glow training data.")


def _process_dataset(input_path: Path, output: Path, backend_name: str, seed: int, crf: int, branch_name: str | None = None) -> tuple[list[dict], Path]:
    videos = find_videos(input_path)
    if not videos:
        raise typer.BadParameter("No supported videos found in input_path")
    output.mkdir(parents=True, exist_ok=True)
    branch = resolve_branch(branch_name) if branch_name else None
    backend = branch.create_backend() if branch else create_backend(backend_name)
    results = []
    try:
        for index, video in enumerate(videos, start=1):
            label = branch.name if branch else backend_name
            typer.echo(f"[{label} {index}/{len(videos)}] {video.name}")
            result = process_video(video, output, backend, seed, crf)
            if branch:
                add_branch_metadata(output / "samples" / video.stem, branch)
            results.append(result.__dict__)
    finally:
        backend.close()
    (output / "manifest.jsonl").write_text("".join(json.dumps(item) + "\n" for item in results), encoding="utf-8")
    return results, build_report(output)


@app.command()
def process(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Argument(...),
    backend_name: str = typer.Option("skin", "--backend"),
    branch: str | None = typer.Option(None, "--branch", help=f"Processing branch: {', '.join(BRANCHES)}."),
    seed: int = typer.Option(7, min=0),
    crf: int = typer.Option(15, min=0, max=51),
) -> None:
    # Old explicit --backend calls retain their original route. With no branch
    # and the untouched default backend, config/pipeline.yaml selects the route.
    selected_branch = branch if branch is not None else (resolve_branch(None).name if backend_name == "skin" else None)
    results, report_path = _process_dataset(input_path, output, backend_name, seed, crf, selected_branch)
    typer.echo(f"Wrote {len(results)} samples and {report_path}")


@app.command()
def compare(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Argument(...),
    backends: str = typer.Option("skin,mediapipe", "--backends", help="Comma-separated backend names."),
    seed: int = typer.Option(7, min=0),
    crf: int = typer.Option(15, min=0, max=51),
) -> None:
    """Run matching source clips through each backend and retain failures for comparison."""
    names = [name.strip().lower() for name in backends.split(",") if name.strip()]
    if not names:
        raise typer.BadParameter("Specify at least one backend")
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in names:
        try:
            results, report_path = _process_dataset(input_path, output / name, name, seed, crf)
            summary.append({"backend": name, "status": "completed", "sample_count": len(results), "report": str(report_path.relative_to(output))})
        except Exception as error:
            typer.echo(f"[{name}] failed: {error}", err=True)
            summary.append({"backend": name, "status": "failed", "error": str(error)})
    summary_path = output / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"Wrote comparison summary: {summary_path}")


@app.command("compare-branches")
def compare_branches(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Argument(...),
    branches: str = typer.Option("classic,refined", "--branches", help=f"Comma-separated branch names: {', '.join(BRANCHES)}."),
    seed: int = typer.Option(7, min=0),
    crf: int = typer.Option(15, min=0, max=51),
) -> None:
    """Run independent classic/refined processing branches on identical clips."""
    names = [name.strip().lower() for name in branches.split(",") if name.strip()]
    if not names:
        raise typer.BadParameter("Specify at least one processing branch")
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in names:
        try:
            selected = resolve_branch(name)
            results, report_path = _process_dataset(input_path, output / selected.name, "mediapipe", seed, crf, selected.name)
            summary.append({"branch": selected.name, "status": "completed", "sample_count": len(results), "report": str(report_path.relative_to(output))})
        except Exception as error:
            typer.echo(f"[{name}] failed: {error}", err=True)
            summary.append({"branch": name, "status": "failed", "error": str(error)})
    summary_path = output / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"Wrote branch comparison summary: {summary_path}")


@app.command()
def report(dataset: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    typer.echo(build_report(dataset))


@app.command("direct-candidates")
def direct_candidates(output: Path = typer.Argument(...)) -> None:
    """Write the current direct-edit model comparison matrix as JSON."""
    typer.echo(candidate_manifest(output))


@app.command("direct")
def direct(
    provider: str = typer.Argument(...),
    source: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Argument(...),
    model: str | None = typer.Option(None, "--model"),
    prompt: str = typer.Option(DEFAULT_GHOST_HAND_PROMPT, "--prompt"),
    seed: int = typer.Option(7, min=0),
    reference_image: Path | None = typer.Option(None, "--reference-image", exists=True, readable=True),
    resolution: str = typer.Option("720P", "--resolution"),
) -> None:
    """Run the selected cloud or local direct video-editing branch."""
    provider = provider.lower()
    try:
        engine = resolve_direct_engine(provider)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if engine == "runway":
        if provider != "runway":
            raise typer.BadParameter("DIRECT_ENGINE=runway requires provider 'runway'.")
        typer.echo(run_runway_video_edit(source, output, model or "gemini_omni_flash", prompt, seed))
    elif engine == "wan":
        if provider not in {"dashscope", "qwen"}:
            raise typer.BadParameter("DIRECT_ENGINE=wan requires provider 'dashscope' or 'qwen'.")
        model = model or "wan2.7-videoedit"
        if model != "wan2.7-videoedit":
            raise typer.BadParameter("DashScope currently supports --model wan2.7-videoedit only.")
        typer.echo(run_dashscope_video_edit(source, output, prompt, seed, resolution, reference_image))
    elif engine == "ltx":
        if provider not in {"ltx", "local"}:
            raise typer.BadParameter("DIRECT_ENGINE=ltx requires provider 'ltx' or 'local'.")
        typer.echo(run_ltx_video_edit(source, output, prompt, seed, reference_image))
    else:
        raise typer.BadParameter("Available providers: ltx, runway, dashscope, qwen")


if __name__ == "__main__":
    app()
