"""`beet` console script — thin dispatch over the Python module surface.

Exposes three shell subcommands, each backed by a Python entry point that
exists today:

  beet gui      -> beet.gui.server.serve (stdlib HTTP, static SPA)
  beet sidecar  -> beet.sidecar.main (stdio JSON-RPC, used by Tauri)
  beet serve    -> uvicorn + beet.api.create_app (requires `beet[api]`)

Analyze / eval / ablation subcommands from the pre-refactor CLI are not
restored here; use `beet gui` (web UI) or the `/evaluation/run` /
`/analyze` endpoints directly until a replacement lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from beet.config import load_config
from beet.pipeline import BeetPipeline

_PROFILE_CHOICES = ["default", "strict", "screening", "no-api", "production"]


@click.group()
def main() -> None:
    """BEET 2.0 — LLM authorship detection."""


@main.command("gui")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8877, show_default=True, type=int)
@click.option("--profile", "-p", default="default",
              type=click.Choice(_PROFILE_CHOICES), show_default=True,
              help="Config profile from configs/<name>.yaml.")
@click.option("--config", "-c", default=None, type=click.Path(exists=True),
              help="Path to a config YAML (overrides --profile).")
@click.option("--no-browser", is_flag=True, help="Don't auto-open the browser.")
def gui(host: str, port: int, profile: str, config: str | None, no_browser: bool) -> None:
    """Run the local web UI (stdlib HTTP, static SPA at /)."""
    from beet.gui.server import serve

    config_path = Path(config) if config else (Path(__file__).parent.parent / "configs" / f"{profile}.yaml")
    cfg = load_config(config_path)
    pipeline = BeetPipeline(cfg)
    serve(
        pipeline,
        host=host, port=port, open_browser=not no_browser,
        config=cfg, profile=profile,
    )


@main.command("sidecar")
@click.option("--profile", "-p", default="default",
              type=click.Choice(_PROFILE_CHOICES), show_default=True)
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--feedback-path", default="data/reviewer_feedback.jsonl", show_default=True)
@click.option("--log-level", default="WARNING",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), show_default=True)
def sidecar_cmd(profile: str, config: str | None, feedback_path: str, log_level: str) -> None:
    """Run the stdio JSON-RPC sidecar (consumed by the Tauri shell)."""
    from beet.sidecar import main as sidecar_main

    argv = [
        "--profile", profile,
        "--feedback-path", feedback_path,
        "--log-level", log_level,
    ]
    if config:
        argv += ["--config", config]
    sys.exit(sidecar_main(argv))


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--profile", "-p", default="default",
              type=click.Choice(_PROFILE_CHOICES), show_default=True)
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
def serve_cmd(host: str, port: int, profile: str, config: str | None) -> None:
    """Run the FastAPI HTTP API (requires `pip install beet[api]`)."""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "`beet serve` requires the [api] extra: `pip install beet[api]`"
        )
    from beet.api import create_app

    config_path = Path(config) if config else (Path(__file__).parent.parent / "configs" / f"{profile}.yaml")
    app = create_app(config_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
