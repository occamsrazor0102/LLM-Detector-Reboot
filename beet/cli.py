import sys
import json
from pathlib import Path
import click
from beet.pipeline import BeetPipeline
from beet.report import build_json_report, build_text_report

@click.group()
def main():
    """BEET 2.0 — LLM Authorship Detection Pipeline"""
    pass

@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default",
              type=click.Choice(["default", "strict", "screening", "no-api"]))
@click.option("--output", "-o", default="text", type=click.Choice(["text", "json"]))
def analyze(text, file, config, profile, output):
    """Analyze a single text for LLM authorship."""
    if file:
        text = Path(file).read_text(encoding="utf-8")
    elif not text:
        text = click.get_text_stream("stdin").read()
    if not text.strip():
        click.echo("Error: no text provided", err=True)
        sys.exit(1)
    config_path = config or (Path(__file__).parent.parent / "configs" / f"{profile}.yaml")
    pipeline = BeetPipeline.from_config_file(config_path)
    determination = pipeline.analyze(text)
    if output == "json":
        click.echo(json.dumps(build_json_report(determination), indent=2))
    else:
        click.echo(build_text_report(determination))
    exit_codes = {"RED": 3, "AMBER": 2, "YELLOW": 1, "GREEN": 0, "UNCERTAIN": 1, "MIXED": 2}
    sys.exit(exit_codes.get(determination.label, 0))

@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default")
@click.option("--text-column", default="text")
@click.option("--output-file", "-o", default=None)
def batch(input_file, config, profile, text_column, output_file):
    """Analyze a batch file (CSV or XLSX)."""
    try:
        import pandas as pd
    except ImportError:
        click.echo("pandas required for batch mode: pip install 'beet[full]'", err=True)
        sys.exit(1)
    suffix = Path(input_file).suffix.lower()
    df = pd.read_excel(input_file) if suffix in (".xlsx", ".xls") else pd.read_csv(input_file)
    if text_column not in df.columns:
        click.echo(f"Column '{text_column}' not found. Available: {list(df.columns)}", err=True)
        sys.exit(1)
    config_path = config or (Path(__file__).parent.parent / "configs" / f"{profile}.yaml")
    pipeline = BeetPipeline.from_config_file(config_path)
    results = []
    with click.progressbar(df.itertuples(), length=len(df), label="Analyzing") as bar:
        for row in bar:
            text = str(getattr(row, text_column, ""))
            det = pipeline.analyze(text)
            results.append({"determination": det.label, "p_llm": round(det.p_llm, 4),
                "ci_lower": round(det.confidence_interval[0], 4), "ci_upper": round(det.confidence_interval[1], 4),
                "detectors_run": "|".join(det.detectors_run), "reason": det.reason})
    out_df = pd.DataFrame(results)
    out_path = output_file or str(Path(input_file).with_suffix(".beet.csv"))
    out_df.to_csv(out_path, index=False)
    click.echo(f"\nResults saved to: {out_path}")
