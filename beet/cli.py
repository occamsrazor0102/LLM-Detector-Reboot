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
@click.option("--id-column", default=None,
              help="Column to use as submission id (default: row index)")
@click.option("--output-file", "-o", default=None)
@click.option("--cross-similarity/--no-cross-similarity", default=False,
              help="Enable cross-submission similarity analysis across the batch")
def batch(input_file, config, profile, text_column, id_column, output_file, cross_similarity):
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

    if cross_similarity:
        from beet.config import load_config
        cfg = load_config(Path(config_path))
        cfg.setdefault("detectors", {}).setdefault("cross_similarity", {})["enabled"] = True
        pipeline = BeetPipeline(cfg)
    else:
        pipeline = BeetPipeline.from_config_file(config_path)

    results = []
    if cross_similarity:
        texts = {}
        for i, row in enumerate(df.itertuples()):
            sid = str(getattr(row, id_column)) if id_column else f"row_{i}"
            texts[sid] = str(getattr(row, text_column, ""))
        click.echo(f"Running batch with cross-similarity across {len(texts)} submissions...")
        det_by_id = pipeline.analyze_batch(texts)
        for sid, det in det_by_id.items():
            results.append({
                "id": sid,
                "determination": det.label, "p_llm": round(det.p_llm, 4),
                "ci_lower": round(det.confidence_interval[0], 4),
                "ci_upper": round(det.confidence_interval[1], 4),
                "detectors_run": "|".join(det.detectors_run),
                "reason": det.reason,
            })
    else:
        with click.progressbar(df.itertuples(), length=len(df), label="Analyzing") as bar:
            for i, row in enumerate(bar):
                text = str(getattr(row, text_column, ""))
                det = pipeline.analyze(text)
                sid = str(getattr(row, id_column)) if id_column else f"row_{i}"
                results.append({
                    "id": sid,
                    "determination": det.label, "p_llm": round(det.p_llm, 4),
                    "ci_lower": round(det.confidence_interval[0], 4),
                    "ci_upper": round(det.confidence_interval[1], 4),
                    "detectors_run": "|".join(det.detectors_run),
                    "reason": det.reason,
                })
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


@main.command("robustness")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default",
              type=click.Choice(["default", "strict", "screening", "no-api"]))
@click.option("--attacks", default="all",
              help="Comma-separated attack names, or 'all' for all transform attacks")
@click.option("--out", "out_file", default=None, type=click.Path())
@click.option("--confirm", is_flag=True, default=False,
              help="Confirm expensive runs (>1000 estimated model calls)")
@click.option("--progress/--no-progress", default=False)
def robustness_cmd(dataset_path, config, profile, attacks, out_file, confirm, progress):
    """Run robustness evaluation across adversarial attacks."""
    from beet.adversarial.registry import list_attacks
    from beet.evaluation import load_dataset, run_robustness_eval
    from dataclasses import asdict

    dataset = load_dataset(dataset_path)
    pipeline = BeetPipeline.from_config_file(_resolve_config_path(config, profile))

    if attacks == "all":
        attack_names = [a.name for a in list_attacks(category="transform")]
    else:
        attack_names = [a.strip() for a in attacks.split(",") if a.strip()]

    n_llm = sum(1 for s in dataset if s.label == 1)
    estimated_calls = len(dataset) + len(attack_names) * (n_llm + sum(1 for s in dataset if s.label == 0))
    if estimated_calls > 1000 and not confirm:
        click.echo(
            f"Refusing: ~{estimated_calls} pipeline calls estimated. Re-run with --confirm.",
            err=True,
        )
        sys.exit(2)

    report = run_robustness_eval(pipeline, dataset, attack_names, progress=progress)

    if out_file:
        payload = {
            "baseline": asdict(report.baseline),
            "per_attack": {k: asdict(v) for k, v in report.per_attack.items()},
            "attack_deltas": report.attack_deltas,
            "vulnerability_ranking": report.vulnerability_ranking,
        }
        Path(out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    click.echo(_format_metrics_block("Baseline:", report.baseline.metrics))
    click.echo("")
    click.echo(f"{'Attack':<24} {'ΔAUROC':>10}  {'ΔECE':>8}  {'ΔBrier':>8}")
    click.echo("-" * 60)
    for name, delta_auroc in report.vulnerability_ranking:
        d = report.attack_deltas[name]
        click.echo(
            f"{name:<24} {delta_auroc:>+10.4f}  "
            f"{d.get('ece', 0.0):>+8.4f}  {d.get('brier', 0.0):>+8.4f}"
        )


@main.command("fairness")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default",
              type=click.Choice(["default", "strict", "screening", "no-api"]))
@click.option("--group-by", "group_key", default="tier",
              help="Sample field to stratify by (tier, source, attack_name)")
@click.option("--threshold", type=float, default=0.50)
@click.option("--max-fpr-ratio", type=float, default=2.0)
@click.option("--out", "out_file", default=None, type=click.Path())
@click.option("--progress/--no-progress", default=False)
def fairness_cmd(dataset_path, config, profile, group_key, threshold, max_fpr_ratio, out_file, progress):
    """Run fairness evaluation stratified by a group key."""
    from beet.evaluation import load_dataset, run_fairness_eval
    from dataclasses import asdict

    dataset = load_dataset(dataset_path)
    pipeline = BeetPipeline.from_config_file(_resolve_config_path(config, profile))
    report = run_fairness_eval(
        pipeline, dataset,
        group_key=group_key, threshold=threshold,
        max_fpr_ratio=max_fpr_ratio, progress=progress,
    )

    if out_file:
        Path(out_file).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    click.echo(f"Overall FPR @ {threshold:.2f}: {report.overall_fpr:.4f}")
    click.echo(f"FPR parity ratio: {report.fpr_parity_ratio:.2f} (alert > {max_fpr_ratio})")
    click.echo("")
    click.echo(f"{'Group':<20} {'n':>6}  {'FPR':>8}  {'ECE':>8}")
    click.echo("-" * 50)
    for g, fpr in sorted(report.per_group_fpr.items()):
        n = report.n_per_group.get(g, 0)
        ece_v = report.per_group_ece.get(g, float("nan"))
        click.echo(f"{g:<20} {n:>6}  {fpr:>8.4f}  {ece_v:>8.4f}")
    for note in report.flagged_disparities:
        click.echo(f"\n[warning] {note}")


@main.group("privacy")
def privacy_grp():
    """Privacy-vault operations."""


@privacy_grp.command("purge")
@click.option("--vault-dir", required=True, type=click.Path())
@click.option("--retention-days", type=int, default=90)
@click.option("--access-log", type=click.Path(), default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--confirm", is_flag=True, default=False)
def privacy_purge(vault_dir, retention_days, access_log, dry_run, confirm):
    """Purge raw-text vault records older than retention-days."""
    from beet.privacy.retention import RetentionManager

    vault = Path(vault_dir)
    if not vault.exists():
        click.echo(f"vault dir does not exist: {vault}", err=True)
        sys.exit(1)

    log_path = Path(access_log) if access_log else None
    rm = RetentionManager(vault_dir=vault, retention_days=retention_days, access_log=log_path)

    if dry_run:
        n = rm.purge_expired(dry_run=True)
        click.echo(f"[dry-run] would purge {n} expired records from {vault}")
        return

    if not confirm:
        click.echo("Refusing destructive purge without --confirm. Re-run with --confirm.", err=True)
        sys.exit(2)

    n = rm.purge_expired(dry_run=False)
    click.echo(f"purged {n} records from {vault}")


@main.command("audit-validate")
@click.argument("log_path", type=click.Path(exists=True))
def audit_validate(log_path):
    """Validate a hash-chained provenance audit log."""
    from beet.provenance.chain import AuditChain

    chain = AuditChain(Path(log_path))
    ok, errors = chain.validate()
    if ok:
        click.echo(f"OK  chain length={len(chain)}  no tampering detected")
    else:
        click.echo(f"FAIL  {len(errors)} error(s):", err=True)
        for e in errors:
            click.echo(f"  - {e}", err=True)
        sys.exit(3)


@main.command("gui")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8877, type=int)
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default")
@click.option("--no-browser", is_flag=True, default=False)
def gui_cmd(host, port, config, profile, no_browser):
    """Launch the embedded web GUI."""
    from beet.gui.server import serve

    pipeline = BeetPipeline.from_config_file(_resolve_config_path(config, profile))
    serve(pipeline, host=host, port=port, open_browser=not no_browser)


@main.command("serve")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
@click.option("--config", "-c", default=None, type=click.Path(exists=True))
@click.option("--profile", "-p", default="default")
def serve_cmd(host, port, config, profile):
    """Run the FastAPI REST server (requires beet[api] extras)."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn required: pip install 'beet[api]'", err=True)
        sys.exit(1)
    from beet.api import create_app

    app = create_app(_resolve_config_path(config, profile))
    uvicorn.run(app, host=host, port=port)
