# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Interactive visualization script for MT evaluation annotations.
Reads .jsonl files containing evaluation results and displays them in an interactive window.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional
from rich.console import Console

# Import tkinter for GUI
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from mt_metrics_eval import data as mt_metrics_eval_data

from mt_evaluation.core import wmt24_lps, wmt23_lps, wmt22_lps, wmt25_lps

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mt_evaluation.data import lang_code2lang
from mt_evaluation.core import Error, HumanEvaluation, Sample, AutomaticEvaluation
from mt_evaluation.meta_evaluation.span_level.standardization import (
    standardize_human_evaluation,
    standardize_text,
)

console = Console()

# Constants for styling and configuration
SEVERITY_LEVELS = ["critical", "major", "minor", "neutral"]
SEVERITY_COLORS = {
    "critical": {"normal": "red", "dark": "darkred"},
    "major": {"normal": "orange", "dark": "darkorange"},
    "minor": {"normal": "goldenrod", "dark": "darkgoldenrod"},
    "neutral": {"normal": "green", "dark": "darkgreen"},
}
SCORE_COLORS = {"good": "green", "medium": "orange", "bad": "red"}


def normalize_severity(severity: str) -> str:
    """Normalize severity to lowercase for consistent handling."""
    return severity.lower() if severity else ""


def get_score_category(score: float) -> str:
    """Get score category based on value."""
    return "good" if score >= -1 else "medium" if score >= -5 else "bad"


def create_error_summary(errors, get_severity_func=lambda e: e.severity):
    """Create error summary counter with normalized severities."""
    return Counter(normalize_severity(get_severity_func(error)) for error in errors)


def format_error_summary(error_summary: Counter) -> str:
    """Format error summary as text."""
    return ", ".join(
        f"{sev}: {count}" for sev, count in error_summary.items() if count > 0
    )


class InteractiveEvaluationBrowser:
    """Interactive GUI for browsing MT evaluation annotations one by one."""

    def __init__(
        self,
        data: List[Sample],
    ):
        """Initialize the interactive browser with evaluation data."""
        self.data = data
        self.current_index = 0
        self.current_font_size = 10
        self.root = tk.Tk()
        self.setup_gui()

    def setup_gui(self):
        """Set up the GUI components."""
        self.root.title("MT Evaluation Browser")
        self.root.geometry("1000x800")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        self._setup_navigation(main_frame)
        self._setup_content_area(main_frame)
        self._configure_text_tags()
        self.display_current_evaluation()

    def _setup_navigation(self, parent):
        """Set up navigation controls."""
        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Navigation buttons
        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_evaluation).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(nav_frame, text="Next ▶", command=self.next_evaluation).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        # Position label
        self.position_label = ttk.Label(nav_frame, text="")
        self.position_label.pack(side=tk.LEFT, padx=(0, 10))

        # Go to specific evaluation
        ttk.Label(nav_frame, text="Go to:").pack(side=tk.LEFT, padx=(0, 5))
        self.goto_entry = ttk.Entry(nav_frame, width=10)
        self.goto_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.goto_entry.bind("<Return>", lambda e: self.goto_evaluation())
        ttk.Button(nav_frame, text="Go", command=self.goto_evaluation).pack(
            side=tk.LEFT, padx=(0, 15)
        )

        # Font controls
        ttk.Label(nav_frame, text="Font:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="A-", command=self.decrease_font_size).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        self.font_size_label = ttk.Label(nav_frame, text=f"{self.current_font_size}")
        self.font_size_label.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(nav_frame, text="A+", command=self.increase_font_size).pack(
            side=tk.LEFT
        )

    def _setup_content_area(self, parent):
        """Set up content display area."""
        content_frame = ttk.Frame(parent)
        content_frame.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S)
        )
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)

        self.text_widget = scrolledtext.ScrolledText(
            content_frame, wrap=tk.WORD, width=80, height=30, font=("Consolas", 10)
        )
        self.text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def _configure_text_tags(self):
        """Configure text styling tags."""
        tag_configs = {
            "title": {"font": ("Consolas", 12, "bold"), "foreground": "blue"},
            "header": {"font": ("Consolas", 10, "bold"), "foreground": "darkgreen"},
            "gold_header": {"font": ("Consolas", 10, "bold"), "foreground": "darkblue"},
        }

        # Add severity-based tags
        for severity in SEVERITY_LEVELS:
            colors = SEVERITY_COLORS[severity]
            tag_configs[f"error_{severity}"] = {
                "foreground": colors["normal"],
                "font": ("Consolas", 10, "bold"),
            }
            tag_configs[f"gold_{severity}"] = {
                "foreground": colors["dark"],
                "font": ("Consolas", 10, "bold"),
            }

        # Add score-based tags
        for category, color in SCORE_COLORS.items():
            tag_configs[f"score_{category}"] = {
                "foreground": color,
                "font": ("Consolas", 10, "bold"),
            }

        # Apply all configurations
        for tag, config in tag_configs.items():
            self.text_widget.tag_configure(tag, **config)

    def display_current_evaluation(self):
        """Display the current evaluation in the text widget."""
        if not self.data:
            return

        self.text_widget.delete(1.0, tk.END)
        record = self.data[self.current_index]
        eval_data = record.evaluation

        # Title and basic info
        self._insert_title_and_header(record, eval_data)

        # Source and target texts
        self._insert_text_section("SOURCE TEXT:", record.src)
        self._insert_text_section("TARGET TEXT:", record.tgt)

        # Model errors
        self._insert_errors_section("MODEL ERRORS:", eval_data.errors, is_gold=False)

        # Gold annotations
        if record.human_evaluation and record.human_evaluation.errors is not None:
            self._insert_errors_section(
                "GOLD ANNOTATIONS (MQM):", record.human_evaluation.errors, is_gold=True
            )
        else:
            self.text_widget.insert(
                tk.END,
                "GOLD ANNOTATIONS: Not available for this sample\n\n",
                "gold_header",
            )

        # Full annotation
        annotation = eval_data.annotation if eval_data.annotation else ""
        if annotation:
            self._insert_text_section("FULL MODEL ANNOTATION:", annotation)

        self.update_position_label()
        self.text_widget.see(1.0)

    def _insert_title_and_header(self, record, eval_data):
        """Insert title and header information."""
        title = (
            f"EVALUATION {self.current_index + 1} of {len(self.data)}\n{'=' * 60}\n\n"
        )
        self.text_widget.insert(tk.END, title, "title")

        score = eval_data.score
        score_tag = f"score_{get_score_category(score)}"
        header_info = (
            f"Language Pair: {record.src_lang} → {record.tgt_lang}\nScore: {score}\n\n"
        )
        self.text_widget.insert(tk.END, header_info, score_tag)

    def _insert_text_section(self, header, text):
        """Insert a text section with header."""
        self.text_widget.insert(tk.END, f"{header}\n", "header")
        self.text_widget.insert(tk.END, f"{text}\n\n")

    def _insert_errors_section(self, header, errors, is_gold=False):
        """Insert errors section with summary and details."""
        tag_prefix = "gold" if is_gold else "error"
        header_tag = "gold_header" if is_gold else "header"

        self.text_widget.insert(tk.END, f"{header}\n", header_tag)

        if errors:
            # Error summary
            if is_gold:
                error_summary = create_error_summary(errors, lambda e: e.severity)
                summary_text = f"Total Gold Errors: {len(errors)} ({format_error_summary(error_summary)})\n\n"
            else:
                error_summary = create_error_summary(errors)
                summary_text = f"Total Model Errors: {len(errors)} ({format_error_summary(error_summary)})\n\n"

            self.text_widget.insert(tk.END, summary_text)

            # Detailed errors
            detail_header = (
                "Detailed Gold Errors:\n" if is_gold else "Detailed Model Errors:\n"
            )
            self.text_widget.insert(tk.END, detail_header)

            for i, error in enumerate(errors, 1):
                if is_gold:
                    self._insert_gold_error_detail(i, error, tag_prefix)
                else:
                    self._insert_model_error_detail(i, error, tag_prefix)
        else:
            no_errors_text = (
                "No gold errors found.\n\n"
                if is_gold
                else "No model errors detected.\n\n"
            )
            self.text_widget.insert(tk.END, no_errors_text)

    def _insert_model_error_detail(self, index, error, tag_prefix):
        """Insert model error details."""
        severity = error.severity
        normalized_severity = normalize_severity(severity)
        category = error.category
        span_text = error.span if error.span else "N/A"
        explanation = error.explanation if error.explanation else ""

        error_tag = f"{tag_prefix}_{normalized_severity}"
        error_text = (
            f"{index:2d}. [{severity.upper()}] {category}\n    Span: {span_text}\n"
        )

        if explanation and explanation != "No explanation provided":
            error_text += f"    Explanation: {explanation}\n"
        error_text += "\n"

        self.text_widget.insert(tk.END, error_text, error_tag)

    def _insert_gold_error_detail(self, index, error, tag_prefix):
        """Insert gold error details."""
        severity = error.severity
        normalized_severity = normalize_severity(severity)
        category = error.category
        span_text = error.span if error.span else "N/A"

        gold_error_tag = f"{tag_prefix}_{normalized_severity}"
        gold_error_text = f"{index:2d}. [GOLD-{severity.upper()}] {category}\n"
        gold_error_text += f"    Span: {span_text}\n    Score: {error.score}\n\n"

        self.text_widget.insert(tk.END, gold_error_text, gold_error_tag)

    def update_position_label(self):
        """Update the position label showing current evaluation number."""
        self.position_label.config(
            text=f"Evaluation {self.current_index + 1} of {len(self.data)}"
        )

    def navigate(self, direction):
        """Navigate to previous/next evaluation."""
        if direction == "prev" and self.current_index > 0:
            self.current_index -= 1
            self.display_current_evaluation()
        elif direction == "next" and self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.display_current_evaluation()
        else:
            boundary = "first" if direction == "prev" else "last"
            messagebox.showinfo("Info", f"Already at the {boundary} evaluation.")

    def previous_evaluation(self):
        """Navigate to the previous evaluation."""
        self.navigate("prev")

    def next_evaluation(self):
        """Navigate to the next evaluation."""
        self.navigate("next")

    def goto_evaluation(self):
        """Go to a specific evaluation number."""
        try:
            target_num = int(self.goto_entry.get())
            if 1 <= target_num <= len(self.data):
                self.current_index = target_num - 1
                self.display_current_evaluation()
                self.goto_entry.delete(0, tk.END)
            else:
                messagebox.showerror(
                    "Error", f"Please enter a number between 1 and {len(self.data)}"
                )
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")

    def adjust_font_size(self, delta):
        """Adjust font size by delta."""
        new_size = max(6, min(24, self.current_font_size + delta))
        if new_size != self.current_font_size:
            self.current_font_size = new_size
            self.update_fonts()

    def increase_font_size(self):
        """Increase the font size."""
        self.adjust_font_size(1)

    def decrease_font_size(self):
        """Decrease the font size."""
        self.adjust_font_size(-1)

    def update_fonts(self):
        """Update all font configurations with the current font size."""
        self.text_widget.configure(font=("Consolas", self.current_font_size))

        title_size = max(self.current_font_size + 2, 8)
        header_size = max(self.current_font_size, 6)

        # Update specific tags
        font_updates = {
            "title": ("Consolas", title_size, "bold"),
            "header": ("Consolas", header_size, "bold"),
            "gold_header": ("Consolas", header_size, "bold"),
        }

        # Add severity and score tags
        for severity in SEVERITY_LEVELS:
            font_updates[f"error_{severity}"] = (
                "Consolas",
                self.current_font_size,
                "bold",
            )
            font_updates[f"gold_{severity}"] = (
                "Consolas",
                self.current_font_size,
                "bold",
            )

        for category in SCORE_COLORS:
            font_updates[f"score_{category}"] = (
                "Consolas",
                self.current_font_size,
                "bold",
            )

        # Apply font updates
        for tag, font in font_updates.items():
            self.text_widget.tag_configure(tag, font=font)

        self.font_size_label.config(text=f"{self.current_font_size}")

    def run(self):
        """Start the interactive browser."""
        self.root.mainloop()


class AnnotationVisualizer:
    """Loads MT evaluation annotations from .jsonl files for interactive visualization."""

    def __init__(
        self,
        test_set: str,
        jsonl_path: str,
        language_pairs: Optional[List[str]] = None,
        included_severities: Optional[List[str]] = None,
        included_categories: Optional[List[str]] = None,
        filter_samples_based_on_gold_errors: bool = False,
    ):
        """Initialize with path to .jsonl file and optional language pair filtering."""
        self.test_set = test_set
        self.jsonl_path = Path(jsonl_path)
        self.language_pairs = language_pairs
        self.included_severities = included_severities or ["minor", "major", "critical"]
        self.included_categories = included_categories or "All"
        self.filter_samples_based_on_gold_errors = filter_samples_based_on_gold_errors
        self.data = []

        self.load_gold_annotations(language_pairs=self.language_pairs)
        self.load_data()
        self.apply_filtering()

    def _extract_system_annotations(self, lp: str, evalset, sys, srcs):
        """Extract annotations for a specific system."""
        tgts = evalset.sys_outputs[sys]
        mqm_ratings = evalset.Ratings("mqm.merged")[sys]

        if not mqm_ratings:
            raise RuntimeError(f"No mqm ratings found for {sys}")

        sys_annotations = []
        for src, tgt, mqm_rating in zip(srcs, tgts, mqm_ratings):
            if mqm_rating is None:
                continue

            errors = [
                Error(
                    span=(
                        src[error.start : error.end]
                        if error.is_source_error
                        else tgt[error.start : error.end]
                    ),
                    category=error.category,
                    severity=error.severity,
                    start=error.start,
                    end=error.end,
                    is_source_error=error.is_source_error,
                    score=-error.score,  # MQM scores are negative
                    explanation=None,
                )
                for error in mqm_rating.errors
            ]

            human_evaluation = HumanEvaluation(
                score=sum(error.score for error in errors), errors=errors
            )

            sys_annotations.append(
                Sample(
                    src=src,
                    tgt=tgt,
                    src_lang=lang_code2lang[lp.split("-")[0]],
                    tgt_lang=lang_code2lang[lp.split("-")[1]],
                    human_evaluation=human_evaluation,
                )
            )

        return sys_annotations

    def load_gold_annotations(self, language_pairs: List[str]):
        """Load gold MQM annotations from mt-metrics-eval for comparison."""

        console.print("[cyan]Loading gold MQM annotations...[/cyan]")

        # Get unique language pairs

        all_annotations = []
        for lp in language_pairs:
            try:
                console.print(f"[cyan]Loading MQM data for {lp}...[/cyan]")
                evalset = mt_metrics_eval_data.EvalSet(
                    self.test_set,
                    lp,
                    read_stored_metric_scores=True,
                    read_stored_ratings=True,
                )

                srcs = evalset.src

                for sys in evalset.sys_outputs:
                    all_annotations += self._extract_system_annotations(
                        lp, evalset, sys, srcs
                    )

            except Exception as e:
                console.print(f"[yellow]Could not load MQM data for {lp}: {e}[/yellow]")

        console.print(
            f"[green]Loaded gold annotations for {len(all_annotations)} samples[/green]"
        )

        self.data = all_annotations

    def load_data(self):
        """Load evaluation data from JSONL file."""
        console.print(f"[cyan]Loading evaluation data from {self.jsonl_path}...[/cyan]")

        if not self.data:
            raise RuntimeError("No gold annotations found.")

        # First pass: collect all data
        all_data = []
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            all_data.append(record)
                        except json.JSONDecodeError as e:
                            console.print(
                                f"[red]Error parsing line {line_num}: {e}[/red]"
                            )

        except FileNotFoundError:
            console.print(f"[red]File not found: {self.jsonl_path}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            sys.exit(1)

        console.print(f"[green]Loaded {len(all_data)} total records[/green]")

        # Create a quick lookup to extract models annotations fast based on human annotations
        data_lookup = {}
        for record in all_data:
            data_lookup[
                record["src"] + record["tgt"] + record["src_lang"] + record["tgt_lang"]
            ] = record["evaluation"]

        # Retain only those samples for which we have a model annotation and human annotations
        remaining_data = []
        for sample in self.data:
            record = data_lookup.get(
                sample.src + sample.tgt + sample.src_lang + sample.tgt_lang, None
            )
            if record is None:
                continue
            sample.evaluation = AutomaticEvaluation.from_dict(record)
            remaining_data.append(sample)

        console.print(
            f"[green]Gold annotations: {len(self.data)}. Intersection with model annotations: {len(remaining_data)}.[/green]"
        )
        self.data = remaining_data

    def apply_filtering(self):
        """Filter samples to show only those containing gold errors matching the specified criteria."""
        if not self.data:
            return

        console.print(
            f"[cyan]Applying filtering - Severities: {self.included_severities}, Categories: {self.included_categories}[/cyan]"
        )

        original_count = len(self.data)
        filtered_data = []

        for sample in self.data:
            if sample.human_evaluation is None:
                continue

            if self.filter_samples_based_on_gold_errors:
                # Check if this sample has at least one gold error matching our criteria
                has_matching_error = False

                for error in sample.human_evaluation.errors:
                    # Check severity
                    severity_match = any(
                        severity_name in error.severity.lower()
                        for severity_name in self.included_severities
                    )

                    # Check category
                    if isinstance(self.included_categories, list):
                        category_match = any(
                            cat.lower() in error.category.lower()
                            for cat in self.included_categories
                        )
                    else:
                        # "All" categories
                        category_match = True

                    if severity_match and category_match:
                        has_matching_error = True
                        break

                if has_matching_error:
                    filtered_data.append(sample)
            else:
                filtered_data.append(sample)

        self.data = filtered_data
        console.print(
            f"[green]Filtered from {original_count} to {len(self.data)} samples containing matching gold errors[/green]"
        )

        if len(self.data) == 0:
            console.print(
                f"[red]No samples found with gold errors matching the specified criteria![/red]"
            )
            console.print(
                f"[yellow]Try using different severity/category filters[/yellow]"
            )

    def _get_lang_name(self, lang_code):
        """Get language name from code."""
        return lang_code2lang.get(lang_code, lang_code)

    def launch_interactive_browser(self):
        """Launch the interactive evaluation browser."""
        console.print(
            "[bold blue]🚀 Launching Interactive Evaluation Browser...[/bold blue]"
        )

        # Test tkinter availability
        try:
            test_root = tk.Tk()
            test_root.withdraw()
            test_root.destroy()
        except Exception as e:
            console.print(f"[red]Tkinter compatibility issue: {e}[/red]")
            console.print(
                "[red]Interactive browser requires a graphical environment[/red]"
            )
            sys.exit(1)

        # Launch GUI
        try:
            browser = InteractiveEvaluationBrowser(self.data)
            browser.run()
        except Exception as e:
            console.print(f"[red]Error launching GUI: {e}[/red]")
            sys.exit(1)


def main():
    """Main function to run the interactive visualization script."""
    parser = argparse.ArgumentParser(
        description="Interactive MT evaluation annotation viewer with filtering options"
    )
    parser.add_argument(
        "jsonl_file", help="Path to the .jsonl file containing evaluations"
    )
    parser.add_argument("--test-set", default="wmt24")
    parser.add_argument("--lps", type=str, nargs="+")
    parser.add_argument(
        "--severities",
        type=str,
        nargs="+",
        default=["minor", "major", "critical"],
        help="Filter to show only samples with gold errors of these severities (default: minor major critical).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default="All",
        help="Filter to show only samples with gold errors of these categories (default: All).",
    )
    parser.add_argument(
        "--filter-samples-based-on-gold-errors",
        action="store_true",
        help="Whether to filter samples based on the characteristics of their gold errors.",
    )

    args = parser.parse_args()

    test_set2lps = {
        "wmt22": wmt22_lps,
        "wmt23": wmt23_lps,
        "wmt24": wmt24_lps,
        "wmt25": wmt25_lps,
    }
    lps = test_set2lps[args.test_set]
    if args.lps:
        lps = args.lps

    # Check if file exists
    if not Path(args.jsonl_file).exists():
        console.print(f"[red]Error: File '{args.jsonl_file}' not found[/red]")
        sys.exit(1)

    # Create visualizer and launch interactive browser
    console.print("[bold cyan]Starting Interactive MT Evaluation Browser[/bold cyan]")

    console.print(f"[cyan]Loading language pairs: {', '.join(lps)}[/cyan]")

    categories = (
        "All"
        if args.categories == "All"
        or (isinstance(args.categories, list) and "All" in args.categories)
        else args.categories
    )

    visualizer = AnnotationVisualizer(
        args.test_set,
        args.jsonl_file,
        language_pairs=lps,
        included_severities=args.severities,
        included_categories=categories,
        filter_samples_based_on_gold_errors=args.filter_samples_based_on_gold_errors,
    )

    try:
        visualizer.launch_interactive_browser()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive browser closed by user.[/yellow]")


if __name__ == "__main__":
    main()
