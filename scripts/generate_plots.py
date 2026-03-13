# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Plot how F1-score varies across progressive span-length levels for each
auto-evaluator.

For every language-pair directory discovered under the given test set(s),
the script reads the corresponding TSV result file and produces a line plot
with progressive-length levels on the x-axis and F1 on the y-axis.

Plots are saved under ``generated/plots/`` mirroring the structure of
``generated/tsv/``.
"""

import argparse
import csv
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend – no GUI required

# Use LaTeX for all text rendering so every label/title/legend is vectorised
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 20,
        "font.size": 16,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PROGRESSIVE_LENGTH_RE = re.compile(r"^(.+?)_PROGRESSIVE_LENGTH_(\d+)$")
RAND_REMOVE_RE = re.compile(r"^(.+?)_RAND_REMOVE_(\d+)$")
REMOVE_ALL_1_RE = re.compile(r"^(.+?)_REMOVE_ALL_1_(\d+)_(\d+)$")

METRIC_TYPE_CHOICES = [
    "exact_match",
    "partial_overlap",
    "character_proportion",
]

CAMERA_READY_METRIC_NAMES_MAP = {
    "exact_match": r"Exact Match (EM)",
    "partial_overlap": r"Match with partial overlap (MP)",
    "character_proportion": r"Macth with partial overlap and partial credit (MPP)",
}

CAMERA_READY_METRIC_NAMES_SHORT_MAP = {
    "exact_match": "EM",
    "partial_overlap": "MP",
    "character_proportion": "MPP",
}

CAMERA_READY_MODEL_NAMES_MAP = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gpt-oss-120b": "gpt-oss 120b",
    "qwen3-235b": "Qwen3 235b",
}

CAMERA_READY_AGGREGATION_NAMES_MAP = {
    "micro": "Micro-averaged",
    "macro": "Macro-averaged",
}


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def read_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot F1-score vs progressive span-length levels for each "
            "auto-evaluator."
        ),
    )
    parser.add_argument(
        "--test-sets",
        type=str,
        nargs="+",
        required=True,
        help="One or more test sets (e.g. wmt25 wmt24).",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        required=True,
        choices=["micro", "macro"],
        help="Aggregation strategy (micro or macro).",
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        choices=["matching", "not_matching"],
        help="Whether severity matching is required.",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        nargs="+",
        default=["exact_match", "partial_overlap", "character_proportion"],
        choices=METRIC_TYPE_CHOICES,
        help="One or more metric types to plot. When multiple are given "
        "they appear in the same plot with different line styles.",
    )
    parser.add_argument(
        "--annotation-protocol",
        type=str,
        default="mqm",
        help="Annotation protocol (default: mqm).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="generated/tsv",
        help="Root directory containing TSV results (default: generated/tsv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated/plots",
        help="Root directory where plots are saved (default: generated/plots).",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Output format(s) for saved plots (default: png).",
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--run-specific-info",
        type=str,
        default="default",
        help=(
            "Additional string to distinguish this run in log messages and "
            "input paths (default: 'default')."
        ),
    )
    return parser


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def load_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a TSV file and return a list of row dicts."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader)


def parse_evaluator_data(
    rows: List[Dict[str, str]],
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Parse TSV rows into a mapping from base evaluator name to a sorted list
    of (progressive_length_level, f1) tuples.

    Evaluators without a ``_PROGRESSIVE_LENGTH_`` suffix are assigned level 0.
    """
    data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        metric_name = row["Metric"]
        f1 = float(row["F1"])
        match = PROGRESSIVE_LENGTH_RE.match(metric_name)
        if match:
            base_name = match.group(1)
            level = int(match.group(2))
        else:
            # Skip RAND_REMOVE and REMOVE_ALL_1 variants – they belong
            # to a different perturbation family.
            if (
                RAND_REMOVE_RE.match(metric_name)
                or REMOVE_ALL_1_RE.match(metric_name)
                or "_REMOVE_ALL_1" in metric_name
            ):
                continue
            base_name = metric_name
            level = 0
        data[base_name].append((level, f1))

    # Sort each evaluator's data by level
    for key in data:
        data[key].sort(key=lambda t: t[0])

    return dict(data)


def parse_rand_remove_data(
    rows: List[Dict[str, str]],
) -> Tuple[
    Dict[str, List[Tuple[int, float]]],
    Dict[str, List[Tuple[float, float]]],
]:
    """Parse TSV rows into rand-remove curves and REMOVE_ALL_1 scatter points.

    Returns
    -------
    rand_remove_data : dict
        ``{base_evaluator: [(pct, f1), ...]}`` where *pct* is the integer
        removal percentage (0 for the unperturbed base evaluator, 10-90 for
        ``_RAND_REMOVE_`` variants).  Sorted by *pct*.
    remove_all_1_data : dict
        ``{base_evaluator: [(effective_pct, f1), ...]}`` for rows matching
        ``_REMOVE_ALL_1_x_y`` where *effective_pct* = 100 * y / x.
    """
    # Collect the set of base evaluators that have RAND_REMOVE variants
    rand_remove_bases: set = set()
    for row in rows:
        m = RAND_REMOVE_RE.match(row["Metric"])
        if m:
            rand_remove_bases.add(m.group(1))

    rand_data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    remove_all_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for row in rows:
        metric_name = row["Metric"]
        f1 = float(row["F1"])

        m_rand = RAND_REMOVE_RE.match(metric_name)
        if m_rand:
            base = m_rand.group(1)
            pct = int(m_rand.group(2))
            rand_data[base].append((pct, f1))
            continue

        m_rem = REMOVE_ALL_1_RE.match(metric_name)
        if m_rem:
            base = m_rem.group(1)
            x = int(m_rem.group(2))
            y = int(m_rem.group(3))
            if x > 0:
                effective_pct = 100.0 * y / x
                remove_all_data[base].append((effective_pct, f1))
            continue

        # Unperturbed base evaluator -> level 0 (only if it has RAND_REMOVE variants, i.e. it is a relevant base evaluator)
        if metric_name in rand_remove_bases:
            rand_data[metric_name].append((0, f1))

    for key in rand_data:
        rand_data[key].sort(key=lambda t: t[0])

    return dict(rand_data), dict(remove_all_data)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def _latex_escape(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    for ch, esc in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(ch, esc)
    return text


def _shorten_labels(names: List[str]) -> List[str]:
    """Strip the longest common prefix shared by all evaluator names."""
    if not names:
        return names
    prefix = os.path.commonprefix(names)
    # Trim to the last underscore so we don't cut mid-word
    idx = prefix.rfind("_")
    if idx > 0:
        prefix = prefix[: idx + 1]
    else:
        prefix = ""
    shortened = [n[len(prefix) :] for n in names]
    
    return [CAMERA_READY_MODEL_NAMES_MAP[name] for name in shortened]


def _save_legend_separately(
    handles,
    labels,
    output_path: Path,
    formats: List[str],
    ) -> None:
    """Render *only* the legend into its own horizontal-strip image file(s)."""
    import math

    # Filter out blank-separator entries so ncol only counts real items
    real_handles, real_labels = zip(
        *[(h, l) for h, l in zip(handles, labels) if l]
    )
    real_handles = list(real_handles)
    real_labels = list(real_labels)

    ncol = min(len(real_labels), 7)
    nrow = math.ceil(len(real_labels) / ncol)

    # Matplotlib fills legends column-first.  Reorder so that column-first
    # reading reproduces the desired row-first order.
    n = len(real_labels)
    reordered_idx = []
    for col in range(ncol):
        for row in range(nrow):
            idx = row * ncol + col
            if idx < n:
                reordered_idx.append(idx)
    real_handles = [real_handles[i] for i in reordered_idx]
    real_labels = [real_labels[i] for i in reordered_idx]

    fig_leg = plt.figure()
    legend = fig_leg.legend(
        real_handles,
        real_labels,
        loc="center",
        ncol=ncol,
        frameon=False,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_leg.savefig(save_path, dpi=150, bbox_inches=bbox)
        logger.info("Saved legend: %s", save_path)
    plt.close(fig_leg)


def plot_progressive_f1(
    metric_evaluator_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
    output_path: Path,
    formats: List[str],
    aggregation: str,
) -> None:
    """
    Create and save a line plot of F1 vs progressive span-length level.

    Parameters
    ----------
    metric_evaluator_data : dict
        Mapping from metric_type -> {evaluator_name -> [(level, f1), ...]}.
        When there is a single metric type the plot uses solid lines only.
        With multiple metric types each type gets a distinct line style.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    metric_types = list(metric_evaluator_data.keys())
    cr_metric_types = [CAMERA_READY_METRIC_NAMES_SHORT_MAP[mt] for mt in metric_types]
    line_styles = ["solid", "dashed", "dotted"]

    # Collect all evaluator names across metric types for consistent colours
    all_evaluators: set = set()
    for ev_data in metric_evaluator_data.values():
        all_evaluators.update(ev_data.keys())
    sorted_names = sorted(all_evaluators)
    short_labels = _shorten_labels(sorted_names)
    label_map = {
        name: _latex_escape(label) for name, label in zip(sorted_names, short_labels)
    }

    markers = ["o"]
    # Assign a consistent colour per evaluator
    cmap = plt.get_cmap("tab10")
    color_map = {name: cmap(i % 10) for i, name in enumerate(sorted_names)}

    multi_metric = len(metric_types) > 1

    # Track which evaluators have already been added to the legend
    legend_evaluators_seen: set = set()

    for mt_idx, metric_type in enumerate(metric_types):
        ev_data = metric_evaluator_data[metric_type]
        ls = line_styles[mt_idx % len(line_styles)]
        for name in sorted_names:
            if name not in ev_data:
                continue
            points = ev_data[name]
            levels = [p[0] for p in points]
            f1s = [p[1] for p in points]
            marker = markers[sorted_names.index(name) % len(markers)]
            # Only label the first occurrence of each evaluator
            if name not in legend_evaluators_seen:
                lbl = label_map[name]
                legend_evaluators_seen.add(name)
            else:
                lbl = None
            ax.plot(
                levels,
                f1s,
                marker=marker,
                label=lbl,
                linewidth=2,
                markersize=6,
                linestyle=ls,
                color=color_map[name],
            )

    cr_aggregation = CAMERA_READY_AGGREGATION_NAMES_MAP[aggregation]
    ax.set_xlabel(r"Span Length (\#additional leading and trailing chars per span)")
    ax.set_ylabel(fr"$F$-Score ({cr_aggregation})")
    # ax.set_title(_latex_escape(title), fontsize=14)

    # Build the legend: evaluator names (colour) + line-style key if multi-metric
    handles, labels = ax.get_legend_handles_labels()
    if multi_metric:
        from matplotlib.lines import Line2D

        # Add a blank separator then line-style entries for each metric type
        handles.append(Line2D([], [], linestyle="none"))
        labels.append("")
        for mt_idx, cr_metric_type in enumerate(cr_metric_types):
            ls = line_styles[mt_idx % len(line_styles)]
            handles.append(Line2D([], [], color="gray", linestyle=ls, linewidth=2))
            labels.append(_latex_escape(cr_metric_type))

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved plot: %s", save_path)

    plt.close(fig)

    # Save legend as a separate image
    legend_path = output_path.parent / (output_path.name + "_legend")
    _save_legend_separately(handles, labels, legend_path, formats)


def plot_rand_remove_f1(
    metric_rand_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
    metric_remove_all_data: Dict[str, Dict[str, List[Tuple[float, float]]]],
    output_path: Path,
    formats: List[str],
    aggregation: str,
) -> None:
    """Create and save a line plot of F1 vs error-removal percentage.

    Parameters
    ----------
    metric_rand_data : dict
        ``{metric_type: {evaluator: [(pct, f1), ...]}}`` -- the RAND_REMOVE
        curves (pct = 0, 10, 20, ..., 90).
    metric_remove_all_data : dict
        ``{metric_type: {evaluator: [(effective_pct, f1), ...]}}`` -- the
        REMOVE_ALL_1 scatter points.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_types = list(metric_rand_data.keys())
    cr_metric_types = [CAMERA_READY_METRIC_NAMES_SHORT_MAP[mt] for mt in metric_types]
    line_styles = ["solid", "dashed", "dotted"]
    scatter_markers = ["x", "+", "*"]

    # Collect all evaluator names across metric types for consistent colours
    all_evaluators: set = set()
    for ev_data in metric_rand_data.values():
        all_evaluators.update(ev_data.keys())
    sorted_names = sorted(all_evaluators)
    short_labels = _shorten_labels(sorted_names)
    label_map = {
        name: _latex_escape(label) for name, label in zip(sorted_names, short_labels)
    }

    markers = ["o"] 
    cmap = plt.get_cmap("tab10")
    color_map = {name: cmap(i % 10) for i, name in enumerate(sorted_names)}

    multi_metric = len(metric_types) > 1
    legend_evaluators_seen: set = set()

    # --- RAND_REMOVE curves ---
    for mt_idx, metric_type in enumerate(metric_types):
        ev_data = metric_rand_data[metric_type]
        ls = line_styles[mt_idx % len(line_styles)]
        for name in sorted_names:
            if name not in ev_data:
                continue
            points = ev_data[name]
            levels = [p[0] for p in points]
            f1s = [p[1] for p in points]
            marker = markers[sorted_names.index(name) % len(markers)]
            if name not in legend_evaluators_seen:
                lbl = label_map[name]
                legend_evaluators_seen.add(name)
            else:
                lbl = None
            ax.plot(
                levels,
                f1s,
                marker=marker,
                label=lbl,
                linewidth=2,
                markersize=6,
                linestyle=ls,
                color=color_map[name],
            )

    # --- REMOVE_ALL_1 scatter points ---
    for mt_idx, metric_type in enumerate(metric_types):
        rem_data = metric_remove_all_data.get(metric_type, {})
        s_marker = scatter_markers[mt_idx % len(scatter_markers)]
        for name in sorted_names:
            if name not in rem_data:
                continue
            for eff_pct, f1 in rem_data[name]:
                ax.scatter(
                    eff_pct,
                    f1,
                    marker=s_marker,
                    s=120,
                    linewidths=2.5,
                    color=color_map[name],
                    zorder=5,
                )

    cr_aggregation = CAMERA_READY_AGGREGATION_NAMES_MAP[aggregation]
    ax.set_xlabel(r"Error removal probability (\%)")
    ax.set_ylabel(rf"$F$-Score ({cr_aggregation})")

    # Build legend
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()

    # Metric-type line-style entries (line only, no marker)
    if multi_metric:
        handles.append(Line2D([], [], linestyle="none"))
        labels.append("")
        for mt_idx, cr_metric_type in enumerate(cr_metric_types):
            ls = line_styles[mt_idx % len(line_styles)]
            handles.append(Line2D([], [], color="gray", linestyle=ls, linewidth=2))
            labels.append(_latex_escape(cr_metric_type))

    # REMOVE_ALL_1 entries – one per metric type with its scatter marker
    handles.append(Line2D([], [], linestyle="none"))
    labels.append("")
    for mt_idx, cr_metric_type in enumerate(cr_metric_types):
        s_marker = scatter_markers[mt_idx % len(scatter_markers)]
        handles.append(
            Line2D(
                [],
                [],
                marker=s_marker,
                color="gray",
                linestyle="none",
                markersize=8,
                markeredgewidth=2.5,
            )
        )
        labels.append(r"Remove-1" + " (" + _latex_escape(cr_metric_type) + ")")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved plot: %s", save_path)

    plt.close(fig)

    # Save legend as a separate image
    legend_path = output_path.parent / (output_path.name + "_legend")
    _save_legend_separately(handles, labels, legend_path, formats)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = read_arguments()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.logging_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    test_sets = args.test_sets
    protocol = args.annotation_protocol
    run_info = args.run_specific_info
    metric_types: List[str] = args.metric_type

    for test_set in test_sets:
        ts_dir = input_root / test_set / protocol / run_info
        if not ts_dir.is_dir():
            logger.warning("Directory not found, skipping: %s", ts_dir)
            continue

        # Discover all language-pair (and global) subdirectories
        lp_dirs = sorted(
            [d for d in ts_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        if not lp_dirs:
            logger.warning("No LP directories found under %s", ts_dir)
            continue

        for lp_dir in lp_dirs:
            # Load data for each requested metric type
            metric_evaluator_data: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
            for mt in metric_types:
                filename = f"{args.aggregation}_{args.matching}_{mt}.tsv"
                tsv_path = lp_dir / filename
                if not tsv_path.is_file():
                    logger.warning("TSV file not found, skipping: %s", tsv_path)
                    continue
                logger.info("Loading %s", tsv_path)
                rows = load_tsv(tsv_path)
                ev_data = parse_evaluator_data(rows)
                if ev_data:
                    metric_evaluator_data[mt] = ev_data

            if not metric_evaluator_data:
                logger.warning("No progressive-length data found for %s", lp_dir)
            else:
                lp_name = lp_dir.name
                title = (
                    f"F-score at increasing span length\n"
                    f"{test_set} | {lp_name}"
                )

                out_stem = f"{args.aggregation}_{args.matching}_{'__'.join(metric_types)}"
                out_path = output_root / test_set / protocol / run_info / lp_name / out_stem

                plot_progressive_f1(metric_evaluator_data, out_path, args.format, args.aggregation)

            # -- RAND_REMOVE plots --
            lp_name = lp_dir.name
            metric_rand_data: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
            metric_remove_all_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
            for mt in metric_types:
                filename = f"{args.aggregation}_{args.matching}_{mt}.tsv"
                tsv_path = lp_dir / filename
                if not tsv_path.is_file():
                    continue
                rows = load_tsv(tsv_path)
                rand_data, rem_data = parse_rand_remove_data(rows)
                if rand_data:
                    metric_rand_data[mt] = rand_data
                if rem_data:
                    metric_remove_all_data[mt] = rem_data

            if metric_rand_data:
                title = (
                    f"F-score at increasing error-removal probability\n"
                    f"{test_set} | {lp_name}"
                )
                out_stem = (
                    f"{args.aggregation}_{args.matching}"
                    f"_{'__'.join(metric_types)}_rand_remove"
                )
                out_path = output_root / test_set / protocol / run_info / lp_name / out_stem
                plot_rand_remove_f1(
                    metric_rand_data,
                    metric_remove_all_data,
                    out_path,
                    args.format,
                    args.aggregation,
                )

    logger.info("Done.")


if __name__ == "__main__":
    main()
