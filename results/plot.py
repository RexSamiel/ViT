#!/usr/bin/env python3
"""
Interactive plot tool for ViT fault injection results.

Usage:
    python plot.py file1.json file2.json file3.json
    python plot.py new_runs/vit_tiny_faulty_12800samples_20260128.json new_runs/deit_tiny_faulty_12800samples_20260128.json

The JSON files are loaded and their names (without path/extension) are used as labels.
Base accuracy is read from each file if available, otherwise falls back to defaults.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons

NUM_BLOCKS = 12
SUBCOMPONENTS = ["qkv", "proj", "fc1", "fc2"]

# Default baseline accuracies (fallback if not in JSON)
DEFAULT_BASELINE_ACC = {
    "vit_tiny": {"top1": 75.77, "top5": 92.69},
    "vit_small": {"top1": 81.39, "top5": 95.74},
    "vit_base": {"top1": 84.53, "top5": 97.20},
    "deit_tiny": {"top1": 72.32, "top5": 91.02},
    "deit_small": {"top1": 79.90, "top5": 95.00},
    "deit_base": {"top1": 81.80, "top5": 95.60},
    "swin_tiny": {"top1": 81.45, "top5": 95.60},
    "swin_small": {"top1": 83.20, "top5": 96.20},
    "swin_base": {"top1": 83.50, "top5": 96.50},
}

# Color palette for up to 10 models
COLOR_PALETTE = [
    {"top1": "#E74C3C", "top5": "#FF6B6B"},  # Red
    {"top1": "#2ECC71", "top5": "#58D68D"},  # Green
    {"top1": "#3498DB", "top5": "#5DADE2"},  # Blue
    {"top1": "#9B59B6", "top5": "#BB8FCE"},  # Purple
    {"top1": "#F39C12", "top5": "#F7DC6F"},  # Orange
    {"top1": "#1ABC9C", "top5": "#48C9B0"},  # Teal
    {"top1": "#E91E63", "top5": "#F48FB1"},  # Pink
    {"top1": "#795548", "top5": "#A1887F"},  # Brown
    {"top1": "#607D8B", "top5": "#90A4AE"},  # Grey
    {"top1": "#FF5722", "top5": "#FF8A65"},  # Deep Orange
]

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]


def load_json(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            return [data]
        return data


def get_label_from_path(path: str) -> str:
    """Extract a clean label from file path."""
    name = Path(path).stem  # filename without extension
    # Remove common prefixes/suffixes for cleaner labels
    for prefix in ["summary_", "results_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def get_model_key_from_path(path: str) -> str:
    """Try to extract model key (vit_tiny, deit_small, etc.) from path."""
    name = Path(path).stem.lower()
    for model_key in DEFAULT_BASELINE_ACC.keys():
        if model_key in name:
            return model_key
    return None


def create_empty_data(models: list[str]) -> dict:
    data = {}
    for model in models:
        data[model] = {}
        for sc in SUBCOMPONENTS:
            data[model][sc] = {
                "top1": {},
                "top5": {},
                "logit_sdc": {},
                "sdc_1pct": {},
                "sdc_5pct": {},
                "sdc_10pct": {},
                "sdc_20pct": {},
                "msdc": {},
                "critical_top1_sdc": {},
                "critical_top5_sdc": {},
            }
    return data


def fill_data(runs: list[dict], model: str, data: dict) -> None:
    for run in runs:
        config = run.get("config", {})
        sc: Optional[str] = config.get("sub_component")
        block: Optional[int] = config.get("block_idx")

        if sc not in SUBCOMPONENTS:
            continue
        if block is None:
            continue

        data[model][sc]["top1"][block] = run.get("avg_top1_acc", run.get("top1_acc", 0))
        data[model][sc]["top5"][block] = run.get("avg_top5_acc", run.get("top5_acc", 0))
        data[model][sc]["logit_sdc"][block] = run.get("avg_logit_sdc", 0)
        data[model][sc]["sdc_1pct"][block] = run.get("avg_sdc_1pct", 0)
        data[model][sc]["sdc_5pct"][block] = run.get("avg_sdc_5pct", 0)
        data[model][sc]["sdc_10pct"][block] = run.get("avg_sdc_10pct", 0)
        data[model][sc]["sdc_20pct"][block] = run.get("avg_sdc_20pct", 0)
        data[model][sc]["msdc"][block] = run.get("avg_msdc", 0)
        data[model][sc]["critical_top1_sdc"][block] = run.get("avg_critical_top1_sdc", 0)
        data[model][sc]["critical_top5_sdc"][block] = run.get("avg_critical_top5_sdc", 0)


def extract_base_accuracy(runs: list[dict], path: str) -> dict:
    """Extract base accuracy from JSON or use defaults."""
    # First check if any run has base_accuracy field
    for run in runs:
        if "base_accuracy" in run:
            base = run["base_accuracy"]
            return {"top1": base.get("top1", 0), "top5": base.get("top5", 0)}

    # Try to get from config model key
    model_key = get_model_key_from_path(path)
    if model_key and model_key in DEFAULT_BASELINE_ACC:
        return DEFAULT_BASELINE_ACC[model_key]

    # Try config.model field
    for run in runs:
        config = run.get("config", {})
        model = config.get("model", "")
        if model in DEFAULT_BASELINE_ACC:
            return DEFAULT_BASELINE_ACC[model]

    # Fallback
    return {"top1": 75.0, "top5": 90.0}


class InteractivePlot:
    def __init__(self, data: dict, models: list[str], colors: dict, baseline_acc: dict):
        self.data = data
        self.models = models
        self.colors = colors
        self.baseline_acc = baseline_acc
        self.show_models = {m: True for m in models}
        self.show_top1 = True
        self.show_top5 = True
        self.plot_mode = "all"
        self.acc_mode = "overall"
        self.data_mode = "accuracy"
        self.view_mode = "pillar"
        self.show_legend = True

        # SDC metric toggles
        self.show_logit_sdc = True
        self.show_sdc_1pct = True
        self.show_sdc_5pct = True
        self.show_sdc_10pct = True
        self.show_sdc_20pct = True
        self.show_critical_top1_sdc = True
        self.show_critical_top5_sdc = True
        self.show_msdc = True

        self.setup_figure()

    def setup_figure(self):
        self.fig = plt.figure(figsize=(20, 10), facecolor="#F8F9FA")

        # Main plot area
        self.main_ax_area = plt.subplot2grid((24, 24), (0, 0), colspan=16, rowspan=24)

        # Control panels
        self.layout_ax = plt.subplot2grid((24, 24), (0, 17), colspan=3, rowspan=6)
        self.model_ax = plt.subplot2grid((24, 24), (6, 17), colspan=3, rowspan=min(len(self.models) + 1, 6))
        self.data_mode_ax = plt.subplot2grid((24, 24), (12, 17), colspan=3, rowspan=2)
        self.view_mode_ax = plt.subplot2grid((24, 24), (14, 17), colspan=3, rowspan=2)
        self.legend_toggle_ax = plt.subplot2grid((24, 24), (16, 17), colspan=3, rowspan=1)

        self.metric_ax = plt.subplot2grid((24, 24), (0, 20), colspan=4, rowspan=2)
        self.acc_ax = plt.subplot2grid((24, 24), (2, 20), colspan=4, rowspan=2)
        self.sdc_metric_ax = plt.subplot2grid((24, 24), (4, 20), colspan=4, rowspan=8)

        self.setup_controls()
        self.update_plot()

    def setup_controls(self):
        # Layout selection
        self.layout_ax.set_title("Plot Layout", fontsize=9, fontweight="bold")
        self.layout_radio = RadioButtons(
            self.layout_ax,
            ("4 Components", "Transformer", "Attention", "MLP", "QKV", "Proj", "FC1", "FC2"),
            active=0,
        )
        self.layout_radio.on_clicked(self.on_layout_change)

        # Model selection - dynamic based on input files
        self.model_ax.set_title("Models", fontsize=10, fontweight="bold")
        self.model_check = CheckButtons(
            self.model_ax,
            self.models,
            [True] * len(self.models)
        )
        self.model_check.on_clicked(self.on_model_change)

        # Data mode selection
        self.data_mode_ax.set_title("Data Mode", fontsize=10, fontweight="bold")
        self.data_mode_radio = RadioButtons(self.data_mode_ax, ("Accuracy", "SDC"), active=0)
        self.data_mode_radio.on_clicked(self.on_data_mode_change)

        # View mode selection
        self.view_mode_ax.set_title("View Mode", fontsize=10, fontweight="bold")
        self.view_mode_radio = RadioButtons(self.view_mode_ax, ("Pillar", "Line"), active=0)
        self.view_mode_radio.on_clicked(self.on_view_mode_change)

        # Legend toggle
        self.legend_toggle_ax.set_title("Options", fontsize=9, fontweight="bold")
        self.legend_toggle_check = CheckButtons(self.legend_toggle_ax, ["Show Legend"], [True])
        self.legend_toggle_check.on_clicked(self.on_legend_toggle)

        # Metric selection
        self.metric_ax.set_title("Accuracy Metrics", fontsize=10, fontweight="bold")
        self.metric_check = CheckButtons(self.metric_ax, ["Top-1", "Top-5"], [True, True])
        self.metric_check.on_clicked(self.on_metric_change)

        # SDC metric selection
        self.sdc_metric_ax.set_title("SDC Metrics", fontsize=9, fontweight="bold")
        self.sdc_metric_check = CheckButtons(
            self.sdc_metric_ax,
            ["Logit SDC", "SDC 1%", "SDC 5%", "SDC 10%", "SDC 20%", "Crit TOP1", "Crit TOP5", "MSDC"],
            [True] * 8,
        )
        self.sdc_metric_check.on_clicked(self.on_sdc_metric_change)

        # Accuracy mode
        self.acc_ax.set_title("Accuracy Mode", fontsize=10, fontweight="bold")
        self.acc_radio = RadioButtons(self.acc_ax, ("Overall Accuracy", "Accuracy Degradation"), active=0)
        self.acc_radio.on_clicked(self.on_acc_mode_change)

    def on_layout_change(self, label):
        layout_map = {
            "4 Components": "all", "Transformer": "transformer", "Attention": "attention",
            "MLP": "mlp", "QKV": "qkv", "Proj": "proj", "FC1": "fc1", "FC2": "fc2",
        }
        self.plot_mode = layout_map[label]
        self.update_plot()

    def on_model_change(self, label):
        self.show_models[label] = not self.show_models[label]
        self.update_plot()

    def on_metric_change(self, label):
        if label == "Top-1":
            self.show_top1 = not self.show_top1
        elif label == "Top-5":
            self.show_top5 = not self.show_top5
        self.update_plot()

    def on_acc_mode_change(self, label):
        self.acc_mode = "overall" if label == "Overall Accuracy" else "degradation"
        self.update_plot()

    def on_data_mode_change(self, label):
        self.data_mode = "accuracy" if label == "Accuracy" else "sdc"
        self.update_plot()

    def on_view_mode_change(self, label):
        self.view_mode = "pillar" if label == "Pillar" else "line"
        self.update_plot()

    def on_legend_toggle(self, label):
        self.show_legend = not self.show_legend
        self.update_plot()

    def on_sdc_metric_change(self, label):
        metric_map = {
            "Logit SDC": "show_logit_sdc", "SDC 1%": "show_sdc_1pct",
            "SDC 5%": "show_sdc_5pct", "SDC 10%": "show_sdc_10pct",
            "SDC 20%": "show_sdc_20pct", "Crit TOP1": "show_critical_top1_sdc",
            "Crit TOP5": "show_critical_top5_sdc", "MSDC": "show_msdc",
        }
        if label in metric_map:
            attr = metric_map[label]
            setattr(self, attr, not getattr(self, attr))
        self.update_plot()

    def calculate_y_limits(self, subcomponents):
        if self.data_mode == "sdc":
            max_val = 0
            for sc in subcomponents:
                for model in self.models:
                    if not self.show_models[model]:
                        continue
                    sdc_metrics = []
                    if self.show_logit_sdc: sdc_metrics.append("logit_sdc")
                    if self.show_sdc_1pct: sdc_metrics.append("sdc_1pct")
                    if self.show_sdc_5pct: sdc_metrics.append("sdc_5pct")
                    if self.show_sdc_10pct: sdc_metrics.append("sdc_10pct")
                    if self.show_sdc_20pct: sdc_metrics.append("sdc_20pct")
                    if self.show_critical_top1_sdc: sdc_metrics.append("critical_top1_sdc")
                    if self.show_critical_top5_sdc: sdc_metrics.append("critical_top5_sdc")
                    if self.show_msdc: sdc_metrics.append("msdc")

                    for metric in sdc_metrics:
                        for value in self.data[model][sc][metric].values():
                            max_val = max(max_val, value)

            max_val = max_val * 1.05
            if max_val < 0.1:
                return 0, max_val
            return 0, max(np.ceil(max_val), 0.1)

        if self.acc_mode == "overall":
            return 0, 100

        max_val = 0
        for sc in subcomponents:
            for model in self.models:
                if not self.show_models[model]:
                    continue
                for metric in ["top1", "top5"]:
                    if metric == "top1" and not self.show_top1: continue
                    if metric == "top5" and not self.show_top5: continue
                    baseline = self.baseline_acc[model][metric]
                    for value in self.data[model][sc][metric].values():
                        max_val = max(max_val, baseline - value)

        max_val = max_val * 1.03
        return 0, max(np.ceil(max_val), 5)

    def get_aggregated_data(self, components, model):
        aggregated = {}
        first_comp = components[0]
        metric_types = list(self.data[model][first_comp].keys())

        for metric_type in metric_types:
            aggregated[metric_type] = {}
            all_blocks = set()
            for comp in components:
                all_blocks.update(self.data[model][comp][metric_type].keys())

            for block_idx in all_blocks:
                values = []
                for comp in components:
                    if block_idx in self.data[model][comp][metric_type]:
                        values.append(self.data[model][comp][metric_type][block_idx])
                if values:
                    aggregated[metric_type][block_idx] = np.mean(values)

        return aggregated

    def update_plot(self):
        for ax in self.fig.axes[:]:
            if ax not in [self.layout_ax, self.model_ax, self.metric_ax, self.acc_ax,
                          self.data_mode_ax, self.view_mode_ax, self.sdc_metric_ax, self.legend_toggle_ax]:
                self.fig.delaxes(ax)

        self.main_ax_area.clear()
        self.main_ax_area.axis("off")

        if self.plot_mode == "all":
            y_min, y_max = self.calculate_y_limits(SUBCOMPONENTS)
            gs = self.main_ax_area.get_subplotspec().subgridspec(2, 2, hspace=0.3, wspace=0.25)
            axes = {
                "qkv": self.fig.add_subplot(gs[0, 0]),
                "proj": self.fig.add_subplot(gs[0, 1]),
                "fc1": self.fig.add_subplot(gs[1, 0]),
                "fc2": self.fig.add_subplot(gs[1, 1]),
            }
            for sc, ax in axes.items():
                self.plot_subcomponent(ax, sc, show_legend=(sc == "qkv" and self.show_legend), y_limits=(y_min, y_max))
        elif self.plot_mode in ["transformer", "attention", "mlp"]:
            if self.plot_mode == "transformer":
                components, title = ["qkv", "proj", "fc1", "fc2"], "Transformer (All Avg)"
            elif self.plot_mode == "attention":
                components, title = ["qkv", "proj"], "Attention (QKV+Proj Avg)"
            else:
                components, title = ["fc1", "fc2"], "MLP (FC1+FC2 Avg)"

            y_min, y_max = self.calculate_y_limits(components)
            gs = self.main_ax_area.get_subplotspec().subgridspec(1, 1)
            ax = self.fig.add_subplot(gs[0, 0])
            self.plot_aggregated(ax, components, title, show_legend=self.show_legend, y_limits=(y_min, y_max))
        else:
            y_min, y_max = self.calculate_y_limits([self.plot_mode])
            gs = self.main_ax_area.get_subplotspec().subgridspec(1, 1)
            ax = self.fig.add_subplot(gs[0, 0])
            self.plot_subcomponent(ax, self.plot_mode, show_legend=self.show_legend, y_limits=(y_min, y_max))

        self.fig.canvas.draw_idle()

    def plot_subcomponent(self, ax, sc: str, show_legend=False, y_limits=None):
        blocks = np.arange(NUM_BLOCKS)
        if self.data_mode == "accuracy":
            self.plot_accuracy_mode(ax, sc, blocks, show_legend, y_limits)
        else:
            self.plot_sdc_mode(ax, sc, blocks, show_legend, y_limits)

    def plot_aggregated(self, ax, components, title, show_legend=False, y_limits=None):
        blocks = np.arange(NUM_BLOCKS)
        if self.data_mode == "accuracy":
            self.plot_aggregated_accuracy(ax, components, title, blocks, show_legend, y_limits)
        else:
            self.plot_aggregated_sdc(ax, components, title, blocks, show_legend, y_limits)

    def plot_accuracy_mode(self, ax, sc: str, blocks, show_legend=False, y_limits=None):
        added_labels = set()
        active_models = [m for m in self.models if self.show_models[m]]
        num_active = len(active_models)

        if num_active == 0 or (not self.show_top1 and not self.show_top5):
            ax.text(0.5, 0.5, "No data selected", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return

        if self.view_mode == "line":
            for model in self.models:
                if not self.show_models[model]:
                    continue
                colors = self.colors[model]

                if self.show_top1:
                    data_dict = self.data[model][sc]["top1"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = []
                        for b in sorted_blocks:
                            val = data_dict[b]
                            if self.acc_mode == "degradation":
                                val = self.baseline_acc[model]["top1"] - val
                            values.append(val)
                        label = f"{model} TOP1" if show_legend else None
                        ax.plot(sorted_blocks, values, color=colors["top1"], marker="o", markersize=6, linewidth=2, label=label, linestyle="-", alpha=0.9)

                if self.show_top5:
                    data_dict = self.data[model][sc]["top5"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = []
                        for b in sorted_blocks:
                            val = data_dict[b]
                            if self.acc_mode == "degradation":
                                val = self.baseline_acc[model]["top5"] - val
                            values.append(val)
                        label = f"{model} TOP5" if show_legend else None
                        ax.plot(sorted_blocks, values, color=colors["top5"], marker="s", markersize=6, linewidth=2, label=label, linestyle="--", alpha=0.9)
        else:
            # Pillar mode
            bar_width = 0.90 / num_active
            start_offset = -0.425
            bar_index = 0

            for model in self.models:
                if not self.show_models[model]:
                    continue
                colors = self.colors[model]
                offset = start_offset + bar_index * bar_width

                if self.show_top1:
                    data_dict = self.data[model][sc]["top1"]
                    if data_dict:
                        label_key = f"{model}_top1"
                        label = f"{model} TOP1" if show_legend and label_key not in added_labels else None
                        if label: added_labels.add(label_key)
                        for block_idx, value in data_dict.items():
                            plot_val = (self.baseline_acc[model]["top1"] - value) if self.acc_mode == "degradation" else value
                            ax.bar(block_idx + offset, plot_val, bar_width, color=colors["top1"],
                                   label=label if block_idx == min(data_dict.keys()) else None,
                                   edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)

                if self.show_top5:
                    data_dict = self.data[model][sc]["top5"]
                    if data_dict:
                        label_key = f"{model}_top5"
                        label = f"{model} TOP5" if show_legend and label_key not in added_labels else None
                        if label: added_labels.add(label_key)
                        for block_idx, value in data_dict.items():
                            plot_val = (self.baseline_acc[model]["top5"] - value) if self.acc_mode == "degradation" else value
                            inner_width = bar_width * 0.7
                            ax.bar(block_idx + offset, plot_val, inner_width, color=colors["top5"],
                                   label=label if block_idx == min(data_dict.keys()) else None,
                                   edgecolor="white", linewidth=0.8, alpha=0.95, zorder=3)

                bar_index += 1

        # Formatting
        ax.set_title(sc.upper(), fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Block Index", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFBFC")

        if y_limits:
            y_min, y_max = y_limits
            ax.set_ylim(y_min, y_max)
            if self.acc_mode == "overall":
                ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
                ax.set_yticks(np.arange(0, 101, 10))
            else:
                ax.set_ylabel("Accuracy Degradation (%)", fontsize=11, fontweight="bold")
                tick_spacing = max(1, int(y_max / 8))
                ax.set_yticks(np.arange(0, y_max + tick_spacing, tick_spacing))

        ax.set_xticks(blocks)
        ax.set_xticklabels(blocks, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_legend and added_labels:
            ax.legend(loc="best", fontsize=9, framealpha=0.95, edgecolor="#BDC3C7", fancybox=True, shadow=True)

    def plot_sdc_mode(self, ax, sc: str, blocks, show_legend=False, y_limits=None):
        added_labels = set()
        active_models = [m for m in self.models if self.show_models[m]]
        num_active = len(active_models)

        has_logit_or_pct = any([self.show_logit_sdc, self.show_sdc_1pct, self.show_sdc_5pct, self.show_sdc_10pct, self.show_sdc_20pct])
        has_critical = self.show_critical_top1_sdc or self.show_critical_top5_sdc
        metric_groups = sum([has_logit_or_pct, has_critical, self.show_msdc])

        if num_active == 0 or metric_groups == 0:
            ax.text(0.5, 0.5, "No data selected", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return

        # SDC colors per model
        def get_sdc_colors(base_color):
            return {
                "logit": base_color, "1pct": base_color, "5pct": base_color,
                "10pct": base_color, "20pct": base_color, "crit1": base_color,
                "crit5": base_color, "msdc": base_color,
            }

        if self.view_mode == "line":
            marker_idx = 0
            for model in self.models:
                if not self.show_models[model]:
                    continue
                color = self.colors[model]["top1"]

                sdc_metrics = []
                if self.show_logit_sdc: sdc_metrics.append(("logit_sdc", "Logit", "-"))
                if self.show_sdc_1pct: sdc_metrics.append(("sdc_1pct", "1%", "--"))
                if self.show_sdc_5pct: sdc_metrics.append(("sdc_5pct", "5%", "-."))
                if self.show_sdc_10pct: sdc_metrics.append(("sdc_10pct", "10%", ":"))
                if self.show_sdc_20pct: sdc_metrics.append(("sdc_20pct", "20%", "-"))
                if self.show_critical_top1_sdc: sdc_metrics.append(("critical_top1_sdc", "Crit T1", "--"))
                if self.show_critical_top5_sdc: sdc_metrics.append(("critical_top5_sdc", "Crit T5", "-."))
                if self.show_msdc: sdc_metrics.append(("msdc", "MSDC", "-"))

                for metric_name, label_suffix, linestyle in sdc_metrics:
                    data_dict = self.data[model][sc][metric_name]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} {label_suffix}" if show_legend else None
                        ax.plot(sorted_blocks, values, color=color, marker=MARKERS[marker_idx % len(MARKERS)],
                                markersize=5, linewidth=2, label=label, linestyle=linestyle, alpha=0.9)
                        marker_idx += 1
        else:
            # Pillar mode for SDC
            total_bars = num_active * metric_groups
            bar_width = 0.90 / total_bars
            start_offset = -0.425
            bar_index = 0

            for model in self.models:
                if not self.show_models[model]:
                    continue
                color = self.colors[model]["top1"]

                if has_logit_or_pct:
                    offset = start_offset + bar_index * bar_width
                    sdc_metrics = []
                    if self.show_logit_sdc: sdc_metrics.append(("logit_sdc", 1.0, "Logit", 2))
                    if self.show_sdc_1pct: sdc_metrics.append(("sdc_1pct", 0.85, "1%", 3))
                    if self.show_sdc_5pct: sdc_metrics.append(("sdc_5pct", 0.70, "5%", 4))
                    if self.show_sdc_10pct: sdc_metrics.append(("sdc_10pct", 0.55, "10%", 5))
                    if self.show_sdc_20pct: sdc_metrics.append(("sdc_20pct", 0.40, "20%", 6))

                    for metric_name, width_mult, label_suffix, z in sdc_metrics:
                        data_dict = self.data[model][sc][metric_name]
                        if data_dict:
                            label_key = f"{model}_{metric_name}"
                            label = f"{model} {label_suffix}" if show_legend and label_key not in added_labels else None
                            if label: added_labels.add(label_key)
                            for block_idx, value in data_dict.items():
                                ax.bar(block_idx + offset, value, bar_width * width_mult, color=color,
                                       label=label if block_idx == min(data_dict.keys()) else None,
                                       edgecolor="white", linewidth=0.8, alpha=0.90, zorder=z)
                    bar_index += 1

                if has_critical:
                    offset = start_offset + bar_index * bar_width
                    if self.show_critical_top5_sdc:
                        data_dict = self.data[model][sc]["critical_top5_sdc"]
                        if data_dict:
                            label_key = f"{model}_crit_top5"
                            label = f"{model} Crit T5" if show_legend and label_key not in added_labels else None
                            if label: added_labels.add(label_key)
                            for block_idx, value in data_dict.items():
                                ax.bar(block_idx + offset, value, bar_width, color=color,
                                       label=label if block_idx == min(data_dict.keys()) else None,
                                       edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)

                    if self.show_critical_top1_sdc:
                        data_dict = self.data[model][sc]["critical_top1_sdc"]
                        if data_dict:
                            label_key = f"{model}_crit_top1"
                            label = f"{model} Crit T1" if show_legend and label_key not in added_labels else None
                            if label: added_labels.add(label_key)
                            for block_idx, value in data_dict.items():
                                ax.bar(block_idx + offset, value, bar_width * 0.7, color=color,
                                       label=label if block_idx == min(data_dict.keys()) else None,
                                       edgecolor="white", linewidth=0.8, alpha=0.95, zorder=3)
                    bar_index += 1

                if self.show_msdc:
                    offset = start_offset + bar_index * bar_width
                    data_dict = self.data[model][sc]["msdc"]
                    if data_dict:
                        label_key = f"{model}_msdc"
                        label = f"{model} MSDC" if show_legend and label_key not in added_labels else None
                        if label: added_labels.add(label_key)
                        for block_idx, value in data_dict.items():
                            ax.bar(block_idx + offset, value, bar_width, color=color,
                                   label=label if block_idx == min(data_dict.keys()) else None,
                                   edgecolor="white", linewidth=0.8, alpha=0.95, zorder=2)
                    bar_index += 1

        # Formatting
        ax.set_title(sc.upper(), fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Block Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("SDC Value (%)", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFBFC")

        if y_limits:
            y_min, y_max = y_limits
            ax.set_ylim(y_min, y_max)
            if y_max < 0.1:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.4f}"))
            elif y_max < 1:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

        ax.set_xticks(blocks)
        ax.set_xticklabels(blocks, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_legend and added_labels:
            ax.legend(loc="best", fontsize=9, framealpha=0.95, edgecolor="#BDC3C7", fancybox=True, shadow=True)

    def plot_aggregated_accuracy(self, ax, components, title, blocks, show_legend=False, y_limits=None):
        # Similar to plot_accuracy_mode but with aggregated data
        added_labels = set()
        active_models = [m for m in self.models if self.show_models[m]]
        num_active = len(active_models)

        if num_active == 0 or (not self.show_top1 and not self.show_top5):
            ax.text(0.5, 0.5, "No data selected", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return

        if self.view_mode == "line":
            for model in self.models:
                if not self.show_models[model]:
                    continue
                agg_data = self.get_aggregated_data(components, model)
                colors = self.colors[model]

                if self.show_top1:
                    data_dict = agg_data["top1"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [(self.baseline_acc[model]["top1"] - data_dict[b]) if self.acc_mode == "degradation" else data_dict[b] for b in sorted_blocks]
                        label = f"{model} TOP1" if show_legend else None
                        ax.plot(sorted_blocks, values, color=colors["top1"], marker="o", markersize=6, linewidth=2, label=label, linestyle="-", alpha=0.9)

                if self.show_top5:
                    data_dict = agg_data["top5"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [(self.baseline_acc[model]["top5"] - data_dict[b]) if self.acc_mode == "degradation" else data_dict[b] for b in sorted_blocks]
                        label = f"{model} TOP5" if show_legend else None
                        ax.plot(sorted_blocks, values, color=colors["top5"], marker="s", markersize=6, linewidth=2, label=label, linestyle="--", alpha=0.9)
        else:
            bar_width = 0.90 / num_active
            start_offset = -0.425
            bar_index = 0

            for model in self.models:
                if not self.show_models[model]:
                    continue
                agg_data = self.get_aggregated_data(components, model)
                colors = self.colors[model]
                offset = start_offset + bar_index * bar_width

                if self.show_top1:
                    data_dict = agg_data["top1"]
                    if data_dict:
                        label_key = f"{model}_top1"
                        label = f"{model} TOP1" if show_legend and label_key not in added_labels else None
                        if label: added_labels.add(label_key)
                        for block_idx, value in data_dict.items():
                            plot_val = (self.baseline_acc[model]["top1"] - value) if self.acc_mode == "degradation" else value
                            ax.bar(block_idx + offset, plot_val, bar_width, color=colors["top1"],
                                   label=label if block_idx == min(data_dict.keys()) else None,
                                   edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)

                if self.show_top5:
                    data_dict = agg_data["top5"]
                    if data_dict:
                        label_key = f"{model}_top5"
                        label = f"{model} TOP5" if show_legend and label_key not in added_labels else None
                        if label: added_labels.add(label_key)
                        for block_idx, value in data_dict.items():
                            plot_val = (self.baseline_acc[model]["top5"] - value) if self.acc_mode == "degradation" else value
                            ax.bar(block_idx + offset, plot_val, bar_width * 0.7, color=colors["top5"],
                                   label=label if block_idx == min(data_dict.keys()) else None,
                                   edgecolor="white", linewidth=0.8, alpha=0.95, zorder=3)

                bar_index += 1

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Block Index", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFBFC")

        if y_limits:
            y_min, y_max = y_limits
            ax.set_ylim(y_min, y_max)
            if self.acc_mode == "overall":
                ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
                ax.set_yticks(np.arange(0, 101, 10))
            else:
                ax.set_ylabel("Accuracy Degradation (%)", fontsize=11, fontweight="bold")

        ax.set_xticks(blocks)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_legend and added_labels:
            ax.legend(loc="best", fontsize=9, framealpha=0.95)

    def plot_aggregated_sdc(self, ax, components, title, blocks, show_legend=False, y_limits=None):
        # Similar to plot_sdc_mode but with aggregated data
        added_labels = set()
        active_models = [m for m in self.models if self.show_models[m]]
        num_active = len(active_models)

        has_logit_or_pct = any([self.show_logit_sdc, self.show_sdc_1pct, self.show_sdc_5pct, self.show_sdc_10pct, self.show_sdc_20pct])
        has_critical = self.show_critical_top1_sdc or self.show_critical_top5_sdc
        metric_groups = sum([has_logit_or_pct, has_critical, self.show_msdc])

        if num_active == 0 or metric_groups == 0:
            ax.text(0.5, 0.5, "No data selected", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return

        total_bars = num_active * metric_groups
        bar_width = 0.90 / total_bars
        start_offset = -0.425
        bar_index = 0

        for model in self.models:
            if not self.show_models[model]:
                continue
            agg_data = self.get_aggregated_data(components, model)
            color = self.colors[model]["top1"]

            if has_logit_or_pct:
                offset = start_offset + bar_index * bar_width
                for metric_name in ["logit_sdc", "sdc_1pct", "sdc_5pct", "sdc_10pct", "sdc_20pct"]:
                    if not getattr(self, f"show_{metric_name.replace('sdc_', 'sdc_')}", True):
                        continue
                    data_dict = agg_data.get(metric_name, {})
                    if data_dict:
                        for block_idx, value in data_dict.items():
                            ax.bar(block_idx + offset, value, bar_width, color=color, edgecolor="white", alpha=0.90, zorder=2)
                bar_index += 1

            if has_critical:
                offset = start_offset + bar_index * bar_width
                for metric_name in ["critical_top1_sdc", "critical_top5_sdc"]:
                    data_dict = agg_data.get(metric_name, {})
                    if data_dict:
                        for block_idx, value in data_dict.items():
                            ax.bar(block_idx + offset, value, bar_width, color=color, edgecolor="white", alpha=0.90, zorder=2)
                bar_index += 1

            if self.show_msdc:
                offset = start_offset + bar_index * bar_width
                data_dict = agg_data.get("msdc", {})
                if data_dict:
                    for block_idx, value in data_dict.items():
                        ax.bar(block_idx + offset, value, bar_width, color=color, edgecolor="white", alpha=0.95, zorder=2)
                bar_index += 1

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Block Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("SDC Value (%)", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFBFC")

        if y_limits:
            ax.set_ylim(y_limits)

        ax.set_xticks(blocks)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive plot tool for ViT fault injection results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot.py file1.json file2.json file3.json
    python plot.py new_runs/*.json
    python plot.py new_runs/vit_tiny*.json new_runs/deit_tiny*.json
        """
    )
    parser.add_argument("files", nargs="+", help="JSON result files to plot")

    args = parser.parse_args()

    if not args.files:
        print("Error: No input files specified")
        print("Usage: python plot.py file1.json file2.json ...")
        sys.exit(1)

    # Validate files exist
    for f in args.files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)

    # Load data from all files
    models = []
    colors = {}
    baseline_acc = {}

    for i, path in enumerate(args.files):
        label = get_label_from_path(path)
        models.append(label)
        colors[label] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        # Load and extract baseline accuracy
        runs = load_json(path)
        baseline_acc[label] = extract_base_accuracy(runs, path)

    # Create data structure
    data = create_empty_data(models)

    # Fill data from each file
    for i, path in enumerate(args.files):
        label = models[i]
        runs = load_json(path)
        fill_data(runs, label, data)

    # Print summary
    print(f"Loaded {len(models)} datasets:")
    for label in models:
        acc = baseline_acc[label]
        print(f"  - {label}: Base accuracy Top-1={acc['top1']:.2f}%, Top-5={acc['top5']:.2f}%")

    # Create interactive plot
    plotter = InteractivePlot(data, models, colors, baseline_acc)
    plt.show()


if __name__ == "__main__":
    main()
