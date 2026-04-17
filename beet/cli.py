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


def _resolve_config_path(config, profile):
    return config or (Path(__file__).parent.parent / "configs" / f"{profile}.yaml")


def _format_metrics_block(title: str, metrics: dict) -> str:
    lines = [title]
    for k in ("auroc", "ece", "brier", "tpr_at_fpr_01"):
        if k in metrics:
            v = metrics[k]
            lines.append(f"  {k:<14} {v:.4f}")
    return "\n".join(lines)


@main.command("eval")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default",
              type=click.Choice(["default", "strict", "screening", "no-api"]))
@click.option("--output", default="text", type=click.Choice(["text", "json"]))
@click.option("--out", "out_file", default=None, type=click.Path(), help="Write JSON report to file")
@click.option("--progress/--no-progress", default=False)
def eval_cmd(dataset_path, config, profile, output, out_file, progress):
    """Run evaluation on a labeled JSONL dataset."""
    from beet.evaluation import load_dataset, run_eval
    from dataclasses import asdict

    dataset = load_dataset(dataset_path)
    pipeline = BeetPipeline.from_config_file(_resolve_config_path(config, profile))
    report = run_eval(pipeline, dataset, progress=progress)

    report_dict = asdict(report)
    if out_file:
        Path(out_file).write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    if output == "json":
        click.echo(json.dumps(report_dict, indent=2))
    else:
        click.echo(f"Samples: {report.n_samples}  (failed: {len(report.failed_samples)})")
        click.echo(f"Config hash: {report.config_hash}")
        click.echo(_format_metrics_block("\nOverall:", report.metrics))
        for tier, m in report.per_tier.items():
            click.echo(_format_metrics_block(f"\n[{tier}]", m))


@main.command("ablation")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default",
              type=click.Choice(["default", "strict", "screening", "no-api"]))
@click.option("--detectors", "detectors_csv", default=None,
              help="Comma-separated list of detectors to ablate (default: all enabled)")
@click.option("--confirm", is_flag=True, default=False,
              help="Confirm expensive runs (>1000 estimated model calls)")
@click.option("--out", "out_file", default=None, type=click.Path())
@click.option("--progress/--no-progress", default=False)
def ablation_cmd(dataset_path, config, profile, detectors_csv, confirm, out_file, progress):
    """Run leave-one-out ablation across detectors."""
    from beet.evaluation import load_dataset, run_ablation, verdict_for
    from beet.config import load_config
    from dataclasses import asdict

    dataset = load_dataset(dataset_path)
    cfg_path = _resolve_config_path(config, profile)
    base_config = load_config(Path(cfg_path))

    targets = None
    if detectors_csv:
        targets = [d.strip() for d in detectors_csv.split(",") if d.strip()]

    enabled = [n for n, c in base_config.get("detectors", {}).items() if c.get("enabled", True)]
    n_ablations = len(targets or enabled)
    estimated_calls = n_ablations * len(dataset)
    if estimated_calls > 1000 and not confirm:
        click.echo(f"Refusing: ~{estimated_calls} pipeline calls estimated. Re-run with --confirm.", err=True)
        sys.exit(2)

    report = run_ablation(base_config, dataset, detectors=targets, progress=progress)

    if out_file:
        Path(out_file).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    click.echo(_format_metrics_block("Baseline:", report.baseline.metrics))
    click.echo("")
    click.echo(f"{'Detector':<24} {'ΔAUROC':>10}  {'ΔECE':>8}  Verdict")
    click.echo("-" * 60)
    for name, _abs in report.ranked:
        d = report.deltas[name]
        click.echo(f"{name:<24} {d['delta_auroc']:>+10.4f}  {d['delta_ece']:>+8.4f}  {verdict_for(d['delta_auroc'])}")
