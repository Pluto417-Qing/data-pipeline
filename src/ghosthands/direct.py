from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlretrieve

try:
    from dotenv import load_dotenv
except ImportError:
    # Segmentation-only commands deliberately work without direct-edit extras.
    load_dotenv = None
else:
    load_dotenv()


DEFAULT_GHOST_HAND_PROMPT = (
    "Preserve the source video exactly: camera, framing, background, hands, fingers, skin, "
    "objects, motion, and timing. Add only a subtle cyan-blue luminous contour attached precisely "
    "to every visible hand. The glow follows each finger and hand edge continuously, with a soft blue "
    "halo and translucent blue hand tint. Do not add hands, change finger count, alter objects, "
    "change the background, or introduce text."
)


@dataclass(frozen=True)
class DirectModelCandidate:
    provider: str
    model: str
    role: str
    notes: str


CANDIDATES = (
    DirectModelCandidate("ltx", "ltxv-2b-0.9.8-distilled-fp8", "local baseline", "Local video-to-video through a ComfyUI API workflow; no per-second API charge, intended for short 480P tests."),
    DirectModelCandidate("runway", "gemini_omni_flash", "fast baseline", "Video-to-video, 720p, source clip up to 10 seconds; use for prompt and reference-image iteration."),
    DirectModelCandidate("runway", "aleph2", "quality baseline", "Video-plus-text/image editing; use a short curated subset to test stronger source preservation."),
    DirectModelCandidate("luma", "ray-2", "independent comparison", "Modify Video with adhere modes; requires the source video at a public URL."),
    DirectModelCandidate("runway", "seedance2_5", "longer creative comparison", "Video-conditioned generation with a larger reference budget; use only after the short-clip baselines."),
    DirectModelCandidate("dashscope", "wan2.7-videoedit", "recommended Qwen-key baseline", "Video editing with a DashScope API key; input 2-10 seconds and optional reference images."),
)


DIRECT_ENGINES = {"auto", "wan", "ltx", "runway"}


def resolve_direct_engine(provider: str) -> str:
    """Resolve the direct-edit branch without changing legacy provider calls."""
    configured = os.getenv("DIRECT_ENGINE", "auto").strip().lower()
    if configured not in DIRECT_ENGINES:
        raise ValueError(f"DIRECT_ENGINE must be one of: {', '.join(sorted(DIRECT_ENGINES))}")
    if configured != "auto":
        return configured
    provider = provider.lower()
    if provider in {"dashscope", "qwen"}:
        return "wan"
    if provider in {"ltx", "local"}:
        return "ltx"
    if provider == "runway":
        return "runway"
    raise ValueError("Available providers: ltx, runway, dashscope, qwen")


def candidate_manifest(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "direct-model-candidates.json"
    path.write_text(json.dumps([asdict(candidate) for candidate in CANDIDATES], indent=2), encoding="utf-8")
    return path


def run_runway_video_edit(source: Path, output: Path, model: str, prompt: str, seed: int) -> Path:
    """Submit one short source clip and save the provider result plus reproducibility metadata."""
    if not os.getenv("RUNWAYML_API_SECRET"):
        raise RuntimeError("RUNWAYML_API_SECRET is required before a paid Runway request can be submitted.")
    if model not in {"gemini_omni_flash", "aleph2", "seedance2_5"}:
        raise ValueError("Runway model must be one of: gemini_omni_flash, aleph2, seedance2_5")
    try:
        from runwayml import RunwayML
    except ImportError as error:
        raise RuntimeError('Install the optional client first: pip install -e ".[runway]"') from error

    output.mkdir(parents=True, exist_ok=True)
    client = RunwayML()
    upload = client.uploads.create_ephemeral(file=source)
    # Seedance uses a different input field from Gemini Omni Flash and Aleph 2.
    request = {"model": model, "prompt_text": prompt, "seed": seed}
    if model == "seedance2_5":
        request.update({"prompt_video": upload.uri, "mode": "edit"})
    else:
        request.update({"video_uri": upload.uri})
    task = client.video_to_video.create(**request).wait_for_task_output()
    task_data = task.model_dump(mode="json") if hasattr(task, "model_dump") else dict(task)
    (output / "provider-task.json").write_text(json.dumps(task_data, indent=2), encoding="utf-8")

    urls = task_data.get("output") or task_data.get("outputs")
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        raise RuntimeError("Runway task completed without an output URL; inspect provider-task.json")
    target = output / "target.mp4"
    urlretrieve(urls[0], target)
    metadata = {
        "provider": "runway",
        "model": model,
        "source": str(source.resolve()),
        "prompt": prompt,
        "task_id": task_data.get("id"),
        "input_uri": upload.uri,
        "seed": seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def _dashscope_upload(source: Path, api_key: str, model: str) -> str:
    """Upload a local input to DashScope's 48-hour temporary storage."""
    import requests

    response = requests.get(
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        params={"action": "getPolicy", "model": model},
        timeout=30,
    )
    response.raise_for_status()
    policy = response.json()["data"]
    key = f"{policy['upload_dir']}/{source.name}"
    with source.open("rb") as handle:
        upload = requests.post(
            policy["upload_host"],
            files={
                "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
                "Signature": (None, policy["signature"]),
                "policy": (None, policy["policy"]),
                "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
                "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
                "key": (None, key),
                "success_action_status": (None, "200"),
                "file": (source.name, handle),
            },
            timeout=300,
        )
    upload.raise_for_status()
    return f"oss://{key}"


def run_dashscope_video_edit(
    source: Path,
    output: Path,
    prompt: str,
    seed: int,
    resolution: str,
    reference_image: Path | None = None,
) -> Path:
    """Edit a local 2-10 second clip with Wan 2.7 and download its target video."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required. Copy .env.example to .env and set the key there.")
    if resolution not in {"720P", "1080P"}:
        raise ValueError("resolution must be 720P or 1080P")
    try:
        from dashscope import VideoSynthesis
    except ImportError as error:
        raise RuntimeError('Install the optional client first: pip install -e ".[dashscope]"') from error

    model = "wan2.7-videoedit"
    output.mkdir(parents=True, exist_ok=True)
    media = [{"type": "video", "url": _dashscope_upload(source, api_key, model)}]
    if reference_image:
        media.append({"type": "reference_image", "url": _dashscope_upload(reference_image, api_key, model)})
    response = VideoSynthesis.call(
        api_key=api_key,
        model=model,
        prompt=prompt,
        media=media,
        resolution=resolution,
        seed=seed,
        prompt_extend=False,
        watermark=False,
        audio_setting="origin",
    )
    if getattr(response, "status_code", None) != 200:
        raise RuntimeError(f"DashScope video edit failed: {getattr(response, 'code', '')} {getattr(response, 'message', '')}")
    task_data = response.to_dict() if hasattr(response, "to_dict") else dict(response)
    (output / "provider-task.json").write_text(json.dumps(task_data, indent=2), encoding="utf-8")
    video_url = response.output.video_url
    target = output / "target.mp4"
    urlretrieve(video_url, target)
    metadata = {
        "provider": "dashscope",
        "model": model,
        "source": str(source.resolve()),
        "reference_image": str(reference_image.resolve()) if reference_image else None,
        "prompt": prompt,
        "seed": seed,
        "resolution": resolution,
        "task_id": getattr(response.output, "task_id", None),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def _replace_ltx_placeholders(value: object, replacements: dict[str, str | int]) -> object:
    if isinstance(value, dict):
        return {key: _replace_ltx_placeholders(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_ltx_placeholders(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _copy_ltx_input(source: Path) -> str:
    input_dir = os.getenv("LTX_COMFYUI_INPUT_DIR")
    if not input_dir:
        raise RuntimeError(
            "LTX_COMFYUI_INPUT_DIR is required for DIRECT_ENGINE=ltx. "
            "Set it to the ComfyUI input directory so its video loader can read the source clip."
        )
    destination_dir = Path(input_dir).expanduser().resolve()
    if not destination_dir.is_dir():
        raise RuntimeError(f"LTX_COMFYUI_INPUT_DIR does not exist: {destination_dir}")
    destination = destination_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination.name


def _download_ltx_result(base_url: str, history: dict, prompt_id: str, output: Path) -> Path:
    import requests

    outputs = history.get(prompt_id, {}).get("outputs", {})
    for node_output in outputs.values():
        for item in node_output.get("gifs", []) + node_output.get("videos", []) + node_output.get("images", []):
            filename = item.get("filename")
            if not filename:
                continue
            response = requests.get(
                f"{base_url}/view",
                params={"filename": filename, "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")},
                timeout=300,
            )
            response.raise_for_status()
            target = output / "target.mp4"
            target.write_bytes(response.content)
            return target
    raise RuntimeError("LTX workflow finished but did not expose a video output. Use a SaveVideo output node.")


def run_ltx_video_edit(
    source: Path,
    output: Path,
    prompt: str,
    seed: int,
    reference_image: Path | None = None,
) -> Path:
    """Run a local LTX video-to-video workflow through ComfyUI's HTTP API."""
    try:
        import requests
    except ImportError as error:
        raise RuntimeError("requests is required for the local LTX branch.") from error

    workflow_setting = os.getenv("LTX_COMFYUI_WORKFLOW")
    if not workflow_setting:
        raise RuntimeError(
            "LTX_COMFYUI_WORKFLOW is required for DIRECT_ENGINE=ltx. Export a working LTX video-to-video "
            "workflow from ComfyUI in API format and set this value to its JSON path."
        )
    workflow_path = Path(workflow_setting).expanduser().resolve()
    if not workflow_path.is_file():
        if "path\\to" in workflow_setting.replace("/", "\\").lower():
            raise RuntimeError(
                "LTX is selected but is not configured yet: LTX_COMFYUI_WORKFLOW still contains the example path. "
                "Install/start ComfyUI with LTX, export its video-to-video workflow in API format, then set the real JSON path."
            )
        raise RuntimeError(f"LTX_COMFYUI_WORKFLOW does not exist: {workflow_path}")
    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"LTX_COMFYUI_WORKFLOW is not valid JSON: {workflow_path}") from error
    if not isinstance(workflow, dict) or "prompt" in workflow:
        raise RuntimeError("LTX_COMFYUI_WORKFLOW must be the API-format node dictionary exported by ComfyUI, not a UI workflow wrapper.")

    base_url = os.getenv("LTX_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    output.mkdir(parents=True, exist_ok=True)
    source_name = _copy_ltx_input(source)
    reference_name = _copy_ltx_input(reference_image) if reference_image else ""
    replacements: dict[str, str | int] = {
        "__SOURCE_VIDEO__": source_name,
        "__REFERENCE_IMAGE__": reference_name,
        "__PROMPT__": prompt,
        "__SEED__": seed,
        "__OUTPUT_PREFIX__": f"ghosthands/{output.name}",
    }
    prepared = _replace_ltx_placeholders(workflow, replacements)
    response = requests.post(f"{base_url}/prompt", json={"prompt": prepared}, timeout=30)
    response.raise_for_status()
    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI accepted the workflow without returning prompt_id.")

    deadline = time.monotonic() + int(os.getenv("LTX_COMFYUI_TIMEOUT_SECONDS", "3600"))
    history: dict = {}
    while time.monotonic() < deadline:
        status = requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
        status.raise_for_status()
        history = status.json()
        if prompt_id in history:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"LTX ComfyUI task timed out after waiting for {prompt_id}.")

    (output / "provider-task.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    target = _download_ltx_result(base_url, history, prompt_id, output)
    metadata = {
        "provider": "local",
        "engine": "ltx",
        "model": os.getenv("LTX_MODEL", "ltxv-2b-0.9.8-distilled-fp8"),
        "source": str(source.resolve()),
        "reference_image": str(reference_image.resolve()) if reference_image else None,
        "prompt": prompt,
        "seed": seed,
        "workflow": str(workflow_path),
        "comfyui_url": base_url,
        "prompt_id": prompt_id,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target
