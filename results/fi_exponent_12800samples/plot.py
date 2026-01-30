import json
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons

NUM_BLOCKS = 12
SUBCOMPONENTS = ["qkv", "proj", "fc1", "fc2"]

FILES = {
    "ViT": "summary_vit_tiny_faulty.json",
    "DeiT": "summary_deit_tiny_faulty.json",
    "Swin": "summary_swin_tiny_faulty.json",
}

# Fault-free baseline accuracies
BASELINE_ACC = {
    "Swin": {"top1": 81.45, "top5": 95.60},
    "DeiT": {"top1": 72.32, "top5": 91.02},
    "ViT": {"top1": 75.77, "top5": 92.69},
}

COLORS = {
    "ViT": {"top1": "#E74C3C", "top5": "#FF6B6B"},
    "DeiT": {"top1": "#2ECC71", "top5": "#58D68D"},
    "Swin": {"top1": "#3498DB", "top5": "#5DADE2"},
}

MARKERS = {
    "ViT": "o",
    "DeiT": "s",
    "Swin": "^",
}


def load_json(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def create_empty_data() -> dict:
    data = {}
    for model in FILES:
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

        data[model][sc]["top1"][block] = run["avg_top1_acc"]
        data[model][sc]["top5"][block] = run["avg_top5_acc"]
        data[model][sc]["logit_sdc"][block] = run.get("avg_logit_sdc", 0)
        data[model][sc]["sdc_1pct"][block] = run.get("avg_sdc_1pct", 0)
        data[model][sc]["sdc_5pct"][block] = run.get("avg_sdc_5pct", 0)
        data[model][sc]["sdc_10pct"][block] = run.get("avg_sdc_10pct", 0)
        data[model][sc]["sdc_20pct"][block] = run.get("avg_sdc_20pct", 0)
        data[model][sc]["msdc"][block] = run.get("avg_msdc", 0)
        data[model][sc]["critical_top1_sdc"][block] = run.get(
            "avg_critical_top1_sdc", 0
        )
        data[model][sc]["critical_top5_sdc"][block] = run.get(
            "avg_critical_top5_sdc", 0
        )


class InteractivePlot:
    def __init__(self, data: dict):
        self.data = data
        self.show_models = {"ViT": True, "DeiT": True, "Swin": True}
        self.show_top1 = True
        self.show_top5 = True
        self.plot_mode = "all"  # "all", "qkv", "proj", "fc1", "fc2"
        self.acc_mode = "overall"  # "overall" or "degradation"
        self.data_mode = "accuracy"  # "accuracy" or "sdc"
        self.view_mode = "pillar"  # "pillar" or "line"
        self.show_legend = True  # Toggle for legend visibility

        # SDC metric toggles - individual controls
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

        # Create main plot area and control panels
        self.main_ax_area = plt.subplot2grid((24, 24), (0, 0), colspan=16, rowspan=24)

        # Control panels - Two column layout
        # Column 1: Plot layout, Models, Data mode, View mode, Legend toggle
        self.layout_ax = plt.subplot2grid((24, 24), (0, 17), colspan=3, rowspan=6)
        self.model_ax = plt.subplot2grid((24, 24), (6, 17), colspan=3, rowspan=3)
        self.data_mode_ax = plt.subplot2grid((24, 24), (9, 17), colspan=3, rowspan=2)
        self.view_mode_ax = plt.subplot2grid((24, 24), (11, 17), colspan=3, rowspan=2)
        self.legend_toggle_ax = plt.subplot2grid(
            (24, 24), (13, 17), colspan=3, rowspan=1
        )

        # Column 2: Accuracy metrics, Accuracy mode, SDC metrics
        self.metric_ax = plt.subplot2grid((24, 24), (0, 20), colspan=4, rowspan=2)
        self.acc_ax = plt.subplot2grid((24, 24), (2, 20), colspan=4, rowspan=2)
        self.sdc_metric_ax = plt.subplot2grid((24, 24), (4, 20), colspan=4, rowspan=8)

        # Setup controls
        self.setup_controls()

        # Initial plot
        self.update_plot()

    def setup_controls(self):
        # Layout selection
        self.layout_ax.set_title("Plot Layout", fontsize=9, fontweight="bold")
        self.layout_radio = RadioButtons(
            self.layout_ax,
            (
                "4 Components",
                "Transformer",
                "Attention",
                "MLP",
                "QKV",
                "Proj",
                "FC1",
                "FC2",
            ),
            active=0,
        )
        self.layout_radio.on_clicked(self.on_layout_change)

        # Model selection
        self.model_ax.set_title("Models", fontsize=10, fontweight="bold")
        self.model_check = CheckButtons(
            self.model_ax, ["ViT", "DeiT", "Swin"], [True, True, True]
        )
        self.model_check.on_clicked(self.on_model_change)

        # Data mode selection
        self.data_mode_ax.set_title("Data Mode", fontsize=10, fontweight="bold")
        self.data_mode_radio = RadioButtons(
            self.data_mode_ax, ("Accuracy", "SDC"), active=0
        )
        self.data_mode_radio.on_clicked(self.on_data_mode_change)

        # View mode selection
        self.view_mode_ax.set_title("View Mode", fontsize=10, fontweight="bold")
        self.view_mode_radio = RadioButtons(
            self.view_mode_ax, ("Pillar", "Line"), active=0
        )
        self.view_mode_radio.on_clicked(self.on_view_mode_change)

        # Legend toggle
        self.legend_toggle_ax.set_title("Options", fontsize=9, fontweight="bold")
        self.legend_toggle_check = CheckButtons(
            self.legend_toggle_ax, ["Show Legend"], [True]
        )
        self.legend_toggle_check.on_clicked(self.on_legend_toggle)

        # Metric selection (for Accuracy mode)
        self.metric_ax.set_title("Accuracy Metrics", fontsize=10, fontweight="bold")
        self.metric_check = CheckButtons(
            self.metric_ax, ["Top-1", "Top-5"], [True, True]
        )
        self.metric_check.on_clicked(self.on_metric_change)

        # SDC metric selection - all in one box
        self.sdc_metric_ax.set_title("SDC Metrics", fontsize=9, fontweight="bold")
        self.sdc_metric_check = CheckButtons(
            self.sdc_metric_ax,
            [
                "Logit SDC",
                "SDC 1%",
                "SDC 5%",
                "SDC 10%",
                "SDC 20%",
                "Crit TOP1",
                "Crit TOP5",
                "MSDC",
            ],
            [True, True, True, True, True, True, True, True],
        )
        self.sdc_metric_check.on_clicked(self.on_sdc_metric_change)

        # Accuracy mode
        self.acc_ax.set_title("Accuracy Mode", fontsize=10, fontweight="bold")
        self.acc_radio = RadioButtons(
            self.acc_ax, ("Overall Accuracy", "Accuracy Degradation"), active=0
        )
        self.acc_radio.on_clicked(self.on_acc_mode_change)

    def on_layout_change(self, label):
        layout_map = {
            "4 Components": "all",
            "Transformer": "transformer",
            "Attention": "attention",
            "MLP": "mlp",
            "QKV": "qkv",
            "Proj": "proj",
            "FC1": "fc1",
            "FC2": "fc2",
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
        if label == "Overall Accuracy":
            self.acc_mode = "overall"
        else:
            self.acc_mode = "degradation"
        self.update_plot()

    def on_data_mode_change(self, label):
        if label == "Accuracy":
            self.data_mode = "accuracy"
        else:
            self.data_mode = "sdc"
        self.update_plot()

    def on_view_mode_change(self, label):
        if label == "Pillar":
            self.view_mode = "pillar"
        else:
            self.view_mode = "line"
        self.update_plot()

    def on_legend_toggle(self, label):
        self.show_legend = not self.show_legend
        self.update_plot()

    def on_sdc_metric_change(self, label):
        if label == "Logit SDC":
            self.show_logit_sdc = not self.show_logit_sdc
        elif label == "SDC 1%":
            self.show_sdc_1pct = not self.show_sdc_1pct
        elif label == "SDC 5%":
            self.show_sdc_5pct = not self.show_sdc_5pct
        elif label == "SDC 10%":
            self.show_sdc_10pct = not self.show_sdc_10pct
        elif label == "SDC 20%":
            self.show_sdc_20pct = not self.show_sdc_20pct
        elif label == "Crit TOP1":
            self.show_critical_top1_sdc = not self.show_critical_top1_sdc
        elif label == "Crit TOP5":
            self.show_critical_top5_sdc = not self.show_critical_top5_sdc
        elif label == "MSDC":
            self.show_msdc = not self.show_msdc
        self.update_plot()

    def calculate_y_limits(self, subcomponents):
        """Calculate consistent y-axis limits across all subcomponents"""
        if self.data_mode == "sdc":
            # For SDC mode, find the max value across all VISIBLE SDC metrics
            max_val = 0
            for sc in subcomponents:
                for model in ["ViT", "DeiT", "Swin"]:
                    if not self.show_models[model]:
                        continue

                    # Check all SDC metrics that are visible
                    sdc_metrics = []
                    if self.show_logit_sdc:
                        sdc_metrics.append("logit_sdc")
                    if self.show_sdc_1pct:
                        sdc_metrics.append("sdc_1pct")
                    if self.show_sdc_5pct:
                        sdc_metrics.append("sdc_5pct")
                    if self.show_sdc_10pct:
                        sdc_metrics.append("sdc_10pct")
                    if self.show_sdc_20pct:
                        sdc_metrics.append("sdc_20pct")
                    if self.show_critical_top1_sdc:
                        sdc_metrics.append("critical_top1_sdc")
                    if self.show_critical_top5_sdc:
                        sdc_metrics.append("critical_top5_sdc")
                    if self.show_msdc:
                        sdc_metrics.append("msdc")

                    for metric in sdc_metrics:
                        data_dict = self.data[model][sc][metric]
                        if not data_dict:
                            continue
                        for value in data_dict.values():
                            max_val = max(max_val, value)

            # Add minimal padding (3%)
            max_val = max_val * 1.05
            # Don't round up for very small values, keep precision
            if max_val < 0.1:
                return 0, max_val
            else:
                max_val = np.ceil(max_val)
                return 0, max(max_val, 0.1)

        # Accuracy mode
        if self.acc_mode == "overall":
            return 0, 100

        # For degradation mode, find the max degradation value
        max_val = 0
        for sc in subcomponents:
            for model in ["ViT", "DeiT", "Swin"]:
                if not self.show_models[model]:
                    continue

                for metric in ["top1", "top5"]:
                    if metric == "top1" and not self.show_top1:
                        continue
                    if metric == "top5" and not self.show_top5:
                        continue

                    data_dict = self.data[model][sc][metric]
                    if not data_dict:
                        continue

                    baseline = BASELINE_ACC[model][metric]
                    for value in data_dict.values():
                        degradation = baseline - value
                        max_val = max(max_val, degradation)

        # Add minimal padding (3%) and round up to nearest integer
        max_val = max_val * 1.03
        max_val = np.ceil(max_val)
        return 0, max(max_val, 5)  # Minimum of 5 for readability

    def get_aggregated_data(self, components, model):
        """Aggregate data across multiple components (average them)"""
        aggregated = {}

        # Get all metric types from the first component
        first_comp = components[0]
        metric_types = list(self.data[model][first_comp].keys())

        for metric_type in metric_types:
            aggregated[metric_type] = {}

            # Find all blocks that exist across all components
            all_blocks = set()
            for comp in components:
                all_blocks.update(self.data[model][comp][metric_type].keys())

            # For each block, average the values across components
            for block_idx in all_blocks:
                values = []
                for comp in components:
                    if block_idx in self.data[model][comp][metric_type]:
                        values.append(self.data[model][comp][metric_type][block_idx])

                if values:
                    aggregated[metric_type][block_idx] = np.mean(values)

        return aggregated

    def update_plot(self):
        # Remove all previous axes in the main area (proper cleanup)
        for ax in self.fig.axes[:]:
            if ax not in [
                self.layout_ax,
                self.model_ax,
                self.metric_ax,
                self.acc_ax,
                self.data_mode_ax,
                self.view_mode_ax,
                self.sdc_metric_ax,
                self.legend_toggle_ax,
            ]:
                self.fig.delaxes(ax)

        # Clear the main area
        self.main_ax_area.clear()
        self.main_ax_area.axis("off")

        if self.plot_mode == "all":
            # Calculate consistent y-limits for all subplots
            y_min, y_max = self.calculate_y_limits(SUBCOMPONENTS)

            # Create 2x2 subplot
            gs = self.main_ax_area.get_subplotspec().subgridspec(
                2, 2, hspace=0.3, wspace=0.25
            )
            axes = {
                "qkv": self.fig.add_subplot(gs[0, 0]),
                "proj": self.fig.add_subplot(gs[0, 1]),
                "fc1": self.fig.add_subplot(gs[1, 0]),
                "fc2": self.fig.add_subplot(gs[1, 1]),
            }
            # Show legend on first subplot only
            for sc, ax in axes.items():
                self.plot_subcomponent(
                    ax,
                    sc,
                    show_legend=(sc == "qkv" and self.show_legend),
                    y_limits=(y_min, y_max),
                )
        elif self.plot_mode in ["transformer", "attention", "mlp"]:
            # Aggregated views
            if self.plot_mode == "transformer":
                components = ["qkv", "proj", "fc1", "fc2"]
                title = "Transformer (All Components Avg)"
            elif self.plot_mode == "attention":
                components = ["qkv", "proj"]
                title = "Attention (QKV + Proj Avg)"
            else:  # mlp
                components = ["fc1", "fc2"]
                title = "MLP (FC1 + FC2 Avg)"

            # Calculate y-limits for aggregated view
            y_min, y_max = self.calculate_y_limits(components)

            gs = self.main_ax_area.get_subplotspec().subgridspec(1, 1)
            ax = self.fig.add_subplot(gs[0, 0])
            self.plot_aggregated(
                ax,
                components,
                title,
                show_legend=self.show_legend,
                y_limits=(y_min, y_max),
            )
        else:
            # Single component subplot
            y_min, y_max = self.calculate_y_limits([self.plot_mode])
            gs = self.main_ax_area.get_subplotspec().subgridspec(1, 1)
            ax = self.fig.add_subplot(gs[0, 0])
            self.plot_subcomponent(
                ax,
                self.plot_mode,
                show_legend=self.show_legend,
                y_limits=(y_min, y_max),
            )

        self.fig.canvas.draw_idle()

    def plot_aggregated(self, ax, components, title, show_legend=False, y_limits=None):
        """Plot aggregated data across multiple components"""
        blocks = np.arange(NUM_BLOCKS)

        if self.data_mode == "accuracy":
            self.plot_aggregated_accuracy(
                ax, components, title, blocks, show_legend, y_limits
            )
        else:
            self.plot_aggregated_sdc(
                ax, components, title, blocks, show_legend, y_limits
            )

    def plot_subcomponent(self, ax, sc: str, show_legend=False, y_limits=None):
        blocks = np.arange(NUM_BLOCKS)

        if self.data_mode == "accuracy":
            self.plot_accuracy_mode(ax, sc, blocks, show_legend, y_limits)
        else:
            self.plot_sdc_mode(ax, sc, blocks, show_legend, y_limits)

    def plot_accuracy_mode(self, ax, sc: str, blocks, show_legend=False, y_limits=None):
        # Track which labels we've added to legend
        added_labels = set()

        # Count active models
        active_models = [m for m in ["ViT", "DeiT", "Swin"] if self.show_models[m]]
        num_active_models = len(active_models)

        if num_active_models == 0 or (not self.show_top1 and not self.show_top5):
            ax.text(
                0.5,
                0.5,
                "No data selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return

        if self.view_mode == "line":
            # LINE MODE
            for model in ["ViT", "DeiT", "Swin"]:
                if not self.show_models[model]:
                    continue

                # Plot top1 line
                if self.show_top1:
                    data_dict = self.data[model][sc]["top1"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = []
                        for block_idx in sorted_blocks:
                            value = data_dict[block_idx]
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top1"]
                                plot_value = baseline - value
                            else:
                                plot_value = value
                            values.append(plot_value)

                        label = f"{model} TOP1" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=COLORS[model]["top1"],
                            marker="o",
                            markersize=6,
                            linewidth=2,
                            label=label,
                            linestyle="-",
                            alpha=0.9,
                        )

                # Plot top5 line
                if self.show_top5:
                    data_dict = self.data[model][sc]["top5"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = []
                        for block_idx in sorted_blocks:
                            value = data_dict[block_idx]
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top5"]
                                plot_value = baseline - value
                            else:
                                plot_value = value
                            values.append(plot_value)

                        label = f"{model} TOP5" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=COLORS[model]["top5"],
                            marker="s",
                            markersize=6,
                            linewidth=2,
                            label=label,
                            linestyle="--",
                            alpha=0.9,
                        )

        else:
            # PILLAR MODE
            total_bars = num_active_models
            bar_width = 0.90 / total_bars
            start_offset = -0.425
            bar_index = 0

            for model in ["ViT", "DeiT", "Swin"]:
                if not self.show_models[model]:
                    continue

                offset = start_offset + bar_index * bar_width

                # Plot top1 FIRST (back layer) if enabled
                if self.show_top1:
                    data_dict = self.data[model][sc]["top1"]
                    if data_dict:
                        label_key = f"{model}_top1"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} TOP1"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top1"]
                                plot_value = baseline - value
                            else:
                                plot_value = value

                            ax.bar(
                                block_idx + offset,
                                plot_value,
                                bar_width,
                                color=COLORS[model]["top1"],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.85,
                                zorder=2,
                            )

                # Plot top5 SECOND (front layer) if enabled
                if self.show_top5:
                    data_dict = self.data[model][sc]["top5"]
                    if data_dict:
                        label_key = f"{model}_top5"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} TOP5"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top5"]
                                plot_value = baseline - value
                            else:
                                plot_value = value

                            inner_width = bar_width * 0.7
                            ax.bar(
                                block_idx + offset,
                                plot_value,
                                inner_width,
                                color=COLORS[model]["top5"],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.95,
                                zorder=3,
                            )

                bar_index += 1

        # Formatting
        ax.set_title(sc.upper(), fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Block Index", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFBFC")

        # Apply y-axis limits
        if y_limits:
            y_min, y_max = y_limits
            ax.set_ylim(y_min, y_max)
            if self.acc_mode == "overall":
                ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
                ax.set_yticks(np.arange(0, 101, 10))
            else:
                ax.set_ylabel(
                    "Accuracy Degradation (%)", fontsize=11, fontweight="bold"
                )
                tick_spacing = max(1, int(y_max / 8))
                ax.set_yticks(np.arange(0, y_max + tick_spacing, tick_spacing))

        ax.set_xticks(blocks)
        ax.set_xticklabels(blocks, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

        # Enhanced grid
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#95A5A6")
        ax.spines["bottom"].set_color("#95A5A6")

        if show_legend and added_labels:
            legend = ax.legend(
                loc="best",
                fontsize=9,
                framealpha=0.95,
                edgecolor="#BDC3C7",
                fancybox=True,
                shadow=True,
            )
            legend.get_frame().set_facecolor("#FFFFFF")

    def plot_sdc_mode(self, ax, sc: str, blocks, show_legend=False, y_limits=None):
        # Track which labels we've added to legend
        added_labels = set()

        # Count active models and metric groups
        active_models = [m for m in ["ViT", "DeiT", "Swin"] if self.show_models[m]]
        num_active_models = len(active_models)

        # Count how many metric groups are active
        metric_groups = 0
        has_logit_or_sdc_pct = (
            self.show_logit_sdc
            or self.show_sdc_1pct
            or self.show_sdc_5pct
            or self.show_sdc_10pct
            or self.show_sdc_20pct
        )
        has_critical = self.show_critical_top1_sdc or self.show_critical_top5_sdc

        if has_logit_or_sdc_pct:
            metric_groups += 1
        if has_critical:
            metric_groups += 1
        if self.show_msdc:
            metric_groups += 1

        if num_active_models == 0 or metric_groups == 0:
            ax.text(
                0.5,
                0.5,
                "No data selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return

        # LINE MODE for SDC
        if self.view_mode == "line":
            sdc_colors = {
                "ViT": {
                    "logit": "#E74C3C",
                    "1pct": "#EC7063",
                    "5pct": "#F1948A",
                    "10pct": "#F5B7B1",
                    "20pct": "#FADBD8",
                    "crit1": "#E74C3C",
                    "crit5": "#F1948A",
                    "msdc": "#C0392B",
                },
                "DeiT": {
                    "logit": "#2ECC71",
                    "1pct": "#52D68D",
                    "5pct": "#76E0A9",
                    "10pct": "#9AEAC5",
                    "20pct": "#BEF4E1",
                    "crit1": "#2ECC71",
                    "crit5": "#76E0A9",
                    "msdc": "#27AE60",
                },
                "Swin": {
                    "logit": "#3498DB",
                    "1pct": "#5DADE2",
                    "5pct": "#85C1E9",
                    "10pct": "#AED6F1",
                    "20pct": "#D6EAF8",
                    "crit1": "#3498DB",
                    "crit5": "#85C1E9",
                    "msdc": "#2874A6",
                },
            }

            markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
            marker_idx = 0

            for model in ["ViT", "DeiT", "Swin"]:
                if not self.show_models[model]:
                    continue

                # Plot each SDC metric as a line
                if self.show_logit_sdc:
                    data_dict = self.data[model][sc]["logit_sdc"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} Logit" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["logit"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle="-",
                            alpha=0.9,
                        )
                        marker_idx += 1

                if self.show_sdc_1pct:
                    data_dict = self.data[model][sc]["sdc_1pct"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} 1%" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["1pct"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle="--",
                            alpha=0.9,
                        )
                        marker_idx += 1

                if self.show_sdc_5pct:
                    data_dict = self.data[model][sc]["sdc_5pct"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} 5%" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["5pct"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle="-.",
                            alpha=0.9,
                        )
                        marker_idx += 1

                if self.show_sdc_10pct:
                    data_dict = self.data[model][sc]["sdc_10pct"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} 10%" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["10pct"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle=":",
                            alpha=0.9,
                        )
                        marker_idx += 1

                if self.show_sdc_20pct:
                    data_dict = self.data[model][sc]["sdc_20pct"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} 20%" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["20pct"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle="-",
                            alpha=0.9,
                        )
                        marker_idx += 1

                if self.show_critical_top1_sdc:
                    data_dict = self.data[model][sc]["critical_top1_sdc"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} Crit T1" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["crit1"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle="--",
                            alpha=0.9,
                        )
                        marker_idx += 1

                if self.show_critical_top5_sdc:
                    data_dict = self.data[model][sc]["critical_top5_sdc"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} Crit T5" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["crit5"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle="-.",
                            alpha=0.9,
                        )
                        marker_idx += 1

                if self.show_msdc:
                    data_dict = self.data[model][sc]["msdc"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = [data_dict[b] for b in sorted_blocks]
                        label = f"{model} MSDC" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=sdc_colors[model]["msdc"],
                            marker=markers[marker_idx % len(markers)],
                            markersize=5,
                            linewidth=2,
                            label=label,
                            linestyle="-",
                            alpha=0.9,
                        )
                        marker_idx += 1

            # Skip to formatting
        else:
            # PILLAR MODE for SDC
            # Calculate bar positions
            total_bars = num_active_models * metric_groups
            bar_width = 0.90 / total_bars
            start_offset = -0.425

            bar_index = 0

            # Define colors for SDC metrics with gradients
            sdc_colors = {
                "ViT": {
                    "logit": "#E74C3C",
                    "1pct": "#EC7063",
                    "5pct": "#F1948A",
                    "10pct": "#F5B7B1",
                    "20pct": "#FADBD8",
                    "crit1": "#E74C3C",
                    "crit5": "#F1948A",
                    "msdc": "#C0392B",
                },
                "DeiT": {
                    "logit": "#2ECC71",
                    "1pct": "#52D68D",
                    "5pct": "#76E0A9",
                    "10pct": "#9AEAC5",
                    "20pct": "#BEF4E1",
                    "crit1": "#2ECC71",
                    "crit5": "#76E0A9",
                    "msdc": "#27AE60",
                },
                "Swin": {
                    "logit": "#3498DB",
                    "1pct": "#5DADE2",
                    "5pct": "#85C1E9",
                    "10pct": "#AED6F1",
                    "20pct": "#D6EAF8",
                    "crit1": "#3498DB",
                    "crit5": "#85C1E9",
                    "msdc": "#2874A6",
                },
            }

            for model in ["ViT", "DeiT", "Swin"]:
                if not self.show_models[model]:
                    continue

                # Plot logit_sdc/sdc_pct group with individual bars in decreasing widths
                if has_logit_or_sdc_pct:
                    offset = start_offset + bar_index * bar_width

                    # Define metrics to plot in order (largest to smallest, back to front)
                    sdc_metrics = []
                    if self.show_logit_sdc:
                        sdc_metrics.append(("logit_sdc", "logit", 1.0, "Logit", 2))
                    if self.show_sdc_1pct:
                        sdc_metrics.append(("sdc_1pct", "1pct", 0.85, "1%", 3))
                    if self.show_sdc_5pct:
                        sdc_metrics.append(("sdc_5pct", "5pct", 0.70, "5%", 4))
                    if self.show_sdc_10pct:
                        sdc_metrics.append(("sdc_10pct", "10pct", 0.55, "10%", 5))
                    if self.show_sdc_20pct:
                        sdc_metrics.append(("sdc_20pct", "20pct", 0.40, "20%", 6))

                    # Plot from back to front (largest to smallest)
                    for (
                        metric_name,
                        color_key,
                        width_mult,
                        label_suffix,
                        z,
                    ) in sdc_metrics:
                        data_dict = self.data[model][sc][metric_name]
                        if data_dict:
                            label_key = f"{model}_{metric_name}"
                            if show_legend and label_key not in added_labels:
                                label = f"{model} SDC {label_suffix}"
                                added_labels.add(label_key)
                            else:
                                label = None

                            for block_idx, value in data_dict.items():
                                ax.bar(
                                    block_idx + offset,
                                    value,
                                    bar_width * width_mult,
                                    color=sdc_colors[model][color_key],
                                    label=label
                                    if block_idx == min(data_dict.keys())
                                    else None,
                                    edgecolor="white",
                                    linewidth=0.8,
                                    alpha=0.90,
                                    zorder=z,
                                )

                    bar_index += 1

                # Plot critical SDC group (critical_top1 behind, critical_top5 in front)
                if has_critical:
                    offset = start_offset + bar_index * bar_width

                    # Plot critical_top5_sdc FIRST (back layer, larger values)
                    if self.show_critical_top5_sdc:
                        data_dict = self.data[model][sc]["critical_top5_sdc"]
                        if data_dict:
                            label_key = f"{model}_crit_top5"
                            if show_legend and label_key not in added_labels:
                                label = f"{model} Crit TOP5"
                                added_labels.add(label_key)
                            else:
                                label = None

                            for block_idx, value in data_dict.items():
                                ax.bar(
                                    block_idx + offset,
                                    value,
                                    bar_width,
                                    color=sdc_colors[model]["crit5"],
                                    label=label
                                    if block_idx == min(data_dict.keys())
                                    else None,
                                    edgecolor="white",
                                    linewidth=0.8,
                                    alpha=0.85,
                                    zorder=2,
                                )

                    # Plot critical_top1_sdc SECOND (front layer, smaller values in front)
                    if self.show_critical_top1_sdc:
                        data_dict = self.data[model][sc]["critical_top1_sdc"]
                        if data_dict:
                            label_key = f"{model}_crit_top1"
                            if show_legend and label_key not in added_labels:
                                label = f"{model} Crit TOP1"
                                added_labels.add(label_key)
                            else:
                                label = None

                            for block_idx, value in data_dict.items():
                                inner_width = bar_width * 0.7
                                ax.bar(
                                    block_idx + offset,
                                    value,
                                    inner_width,
                                    color=sdc_colors[model]["crit1"],
                                    label=label
                                    if block_idx == min(data_dict.keys())
                                    else None,
                                    edgecolor="white",
                                    linewidth=0.8,
                                    alpha=0.95,
                                    zorder=3,
                                )

                    bar_index += 1

                # Plot MSDC (separate, not nested)
                if self.show_msdc:
                    offset = start_offset + bar_index * bar_width

                    data_dict = self.data[model][sc]["msdc"]
                    if data_dict:
                        label_key = f"{model}_msdc"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} MSDC"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            ax.bar(
                                block_idx + offset,
                                value,
                                bar_width,
                                color=sdc_colors[model]["msdc"],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.95,
                                zorder=2,
                            )

                    bar_index += 1

        # Formatting
        ax.set_title(sc.upper(), fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Block Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("SDC Value (%)", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFBFC")

        # Apply y-axis limits
        if y_limits:
            y_min, y_max = y_limits
            ax.set_ylim(y_min, y_max)
            # Smart tick spacing for both large and small values
            if y_max < 0.01:
                # For very very small values (< 0.01), use scientific notation
                tick_spacing = y_max / 8
                ticks = np.arange(0, y_max + tick_spacing, tick_spacing)
                ax.set_yticks(ticks)
                ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
            elif y_max < 0.1:
                # For very small values (0.01-0.1), use 3-4 decimal places
                tick_spacing = y_max / 8
                ticks = np.arange(0, y_max + tick_spacing, tick_spacing)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.4f}"))
            elif y_max < 1:
                # For small values (0.1-1), use 2 decimal places
                tick_spacing = y_max / 8
                ticks = np.arange(0, y_max + tick_spacing, tick_spacing)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
            elif y_max < 10:
                # For small values (1-10), use 1 decimal place
                tick_spacing = max(0.5, y_max / 8)
                ax.set_yticks(np.arange(0, y_max + tick_spacing, tick_spacing))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
            else:
                # For larger values, use integer spacing
                tick_spacing = max(1, int(y_max / 8))
                ax.set_yticks(np.arange(0, y_max + tick_spacing, tick_spacing))

        ax.set_xticks(blocks)
        ax.set_xticklabels(blocks, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

        # Enhanced grid
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#95A5A6")
        ax.spines["bottom"].set_color("#95A5A6")

        if show_legend and added_labels:
            legend = ax.legend(
                loc="best",
                fontsize=9,
                framealpha=0.95,
                edgecolor="#BDC3C7",
                fancybox=True,
                shadow=True,
            )
            legend.get_frame().set_facecolor("#FFFFFF")

    def plot_aggregated_accuracy(
        self, ax, components, title, blocks, show_legend=False, y_limits=None
    ):
        """Plot aggregated accuracy data across multiple components"""
        added_labels = set()

        active_models = [m for m in ["ViT", "DeiT", "Swin"] if self.show_models[m]]
        num_active_models = len(active_models)

        if num_active_models == 0 or (not self.show_top1 and not self.show_top5):
            ax.text(
                0.5,
                0.5,
                "No data selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return

        if self.view_mode == "line":
            # LINE MODE for aggregated
            for model in ["ViT", "DeiT", "Swin"]:
                if not self.show_models[model]:
                    continue

                agg_data = self.get_aggregated_data(components, model)

                # Plot top1 line
                if self.show_top1:
                    data_dict = agg_data["top1"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = []
                        for block_idx in sorted_blocks:
                            value = data_dict[block_idx]
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top1"]
                                plot_value = baseline - value
                            else:
                                plot_value = value
                            values.append(plot_value)

                        label = f"{model} TOP1" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=COLORS[model]["top1"],
                            marker="o",
                            markersize=6,
                            linewidth=2,
                            label=label,
                            linestyle="-",
                            alpha=0.9,
                        )

                # Plot top5 line
                if self.show_top5:
                    data_dict = agg_data["top5"]
                    if data_dict:
                        sorted_blocks = sorted(data_dict.keys())
                        values = []
                        for block_idx in sorted_blocks:
                            value = data_dict[block_idx]
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top5"]
                                plot_value = baseline - value
                            else:
                                plot_value = value
                            values.append(plot_value)

                        label = f"{model} TOP5" if show_legend else None
                        ax.plot(
                            sorted_blocks,
                            values,
                            color=COLORS[model]["top5"],
                            marker="s",
                            markersize=6,
                            linewidth=2,
                            label=label,
                            linestyle="--",
                            alpha=0.9,
                        )

        else:
            # PILLAR MODE for aggregated
            total_bars = num_active_models
            bar_width = 0.90 / total_bars
            start_offset = -0.425
            bar_index = 0

            for model in ["ViT", "DeiT", "Swin"]:
                if not self.show_models[model]:
                    continue

                offset = start_offset + bar_index * bar_width

                # Get aggregated data for this model
                agg_data = self.get_aggregated_data(components, model)

                # Plot top1 FIRST (back layer)
                if self.show_top1:
                    data_dict = agg_data["top1"]
                    if data_dict:
                        label_key = f"{model}_top1"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} TOP1"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top1"]
                                plot_value = baseline - value
                            else:
                                plot_value = value

                            ax.bar(
                                block_idx + offset,
                                plot_value,
                                bar_width,
                                color=COLORS[model]["top1"],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.85,
                                zorder=2,
                            )

                # Plot top5 SECOND (front layer)
                if self.show_top5:
                    data_dict = agg_data["top5"]
                    if data_dict:
                        label_key = f"{model}_top5"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} TOP5"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            if self.acc_mode == "degradation":
                                baseline = BASELINE_ACC[model]["top5"]
                                plot_value = baseline - value
                            else:
                                plot_value = value

                            inner_width = bar_width * 0.7
                            ax.bar(
                                block_idx + offset,
                                plot_value,
                                inner_width,
                                color=COLORS[model]["top5"],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.95,
                                zorder=3,
                            )

                bar_index += 1

        # Formatting
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
                ax.set_ylabel(
                    "Accuracy Degradation (%)", fontsize=11, fontweight="bold"
                )
                tick_spacing = max(1, int(y_max / 8))
                ax.set_yticks(np.arange(0, y_max + tick_spacing, tick_spacing))

        ax.set_xticks(blocks)
        ax.set_xticklabels(blocks, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#95A5A6")
        ax.spines["bottom"].set_color("#95A5A6")

        if show_legend and added_labels:
            legend = ax.legend(
                loc="best",
                fontsize=9,
                framealpha=0.95,
                edgecolor="#BDC3C7",
                fancybox=True,
                shadow=True,
            )
            legend.get_frame().set_facecolor("#FFFFFF")

    def plot_aggregated_sdc(
        self, ax, components, title, blocks, show_legend=False, y_limits=None
    ):
        """Plot aggregated SDC data across multiple components"""
        added_labels = set()

        active_models = [m for m in ["ViT", "DeiT", "Swin"] if self.show_models[m]]
        num_active_models = len(active_models)

        metric_groups = 0
        has_logit_or_sdc_pct = (
            self.show_logit_sdc
            or self.show_sdc_1pct
            or self.show_sdc_5pct
            or self.show_sdc_10pct
            or self.show_sdc_20pct
        )
        has_critical = self.show_critical_top1_sdc or self.show_critical_top5_sdc

        if has_logit_or_sdc_pct:
            metric_groups += 1
        if has_critical:
            metric_groups += 1
        if self.show_msdc:
            metric_groups += 1

        if num_active_models == 0 or metric_groups == 0:
            ax.text(
                0.5,
                0.5,
                "No data selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return

        total_bars = num_active_models * metric_groups
        bar_width = 0.90 / total_bars
        start_offset = -0.425
        bar_index = 0

        sdc_colors = {
            "ViT": {
                "logit": "#E74C3C",
                "1pct": "#EC7063",
                "5pct": "#F1948A",
                "10pct": "#F5B7B1",
                "20pct": "#FADBD8",
                "crit1": "#E74C3C",
                "crit5": "#F1948A",
                "msdc": "#C0392B",
            },
            "DeiT": {
                "logit": "#2ECC71",
                "1pct": "#52D68D",
                "5pct": "#76E0A9",
                "10pct": "#9AEAC5",
                "20pct": "#BEF4E1",
                "crit1": "#2ECC71",
                "crit5": "#76E0A9",
                "msdc": "#27AE60",
            },
            "Swin": {
                "logit": "#3498DB",
                "1pct": "#5DADE2",
                "5pct": "#85C1E9",
                "10pct": "#AED6F1",
                "20pct": "#D6EAF8",
                "crit1": "#3498DB",
                "crit5": "#85C1E9",
                "msdc": "#2874A6",
            },
        }

        for model in ["ViT", "DeiT", "Swin"]:
            if not self.show_models[model]:
                continue

            # Get aggregated data for this model
            agg_data = self.get_aggregated_data(components, model)

            # Plot logit_sdc/sdc_pct group
            if has_logit_or_sdc_pct:
                offset = start_offset + bar_index * bar_width

                sdc_metrics = []
                if self.show_logit_sdc:
                    sdc_metrics.append(("logit_sdc", "logit", 1.0, "Logit", 2))
                if self.show_sdc_1pct:
                    sdc_metrics.append(("sdc_1pct", "1pct", 0.85, "1%", 3))
                if self.show_sdc_5pct:
                    sdc_metrics.append(("sdc_5pct", "5pct", 0.70, "5%", 4))
                if self.show_sdc_10pct:
                    sdc_metrics.append(("sdc_10pct", "10pct", 0.55, "10%", 5))
                if self.show_sdc_20pct:
                    sdc_metrics.append(("sdc_20pct", "20pct", 0.40, "20%", 6))

                for metric_name, color_key, width_mult, label_suffix, z in sdc_metrics:
                    data_dict = agg_data[metric_name]
                    if data_dict:
                        label_key = f"{model}_{metric_name}"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} SDC {label_suffix}"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            ax.bar(
                                block_idx + offset,
                                value,
                                bar_width * width_mult,
                                color=sdc_colors[model][color_key],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.90,
                                zorder=z,
                            )

                bar_index += 1

            # Plot critical SDC group
            if has_critical:
                offset = start_offset + bar_index * bar_width

                # Critical TOP5 first (back layer)
                if self.show_critical_top5_sdc:
                    data_dict = agg_data["critical_top5_sdc"]
                    if data_dict:
                        label_key = f"{model}_crit_top5"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} Crit TOP5"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            ax.bar(
                                block_idx + offset,
                                value,
                                bar_width,
                                color=sdc_colors[model]["crit5"],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.85,
                                zorder=2,
                            )

                # Critical TOP1 second (front layer)
                if self.show_critical_top1_sdc:
                    data_dict = agg_data["critical_top1_sdc"]
                    if data_dict:
                        label_key = f"{model}_crit_top1"
                        if show_legend and label_key not in added_labels:
                            label = f"{model} Crit TOP1"
                            added_labels.add(label_key)
                        else:
                            label = None

                        for block_idx, value in data_dict.items():
                            inner_width = bar_width * 0.7
                            ax.bar(
                                block_idx + offset,
                                value,
                                inner_width,
                                color=sdc_colors[model]["crit1"],
                                label=label
                                if block_idx == min(data_dict.keys())
                                else None,
                                edgecolor="white",
                                linewidth=0.8,
                                alpha=0.95,
                                zorder=3,
                            )

                bar_index += 1

            # Plot MSDC
            if self.show_msdc:
                offset = start_offset + bar_index * bar_width
                data_dict = agg_data["msdc"]
                if data_dict:
                    label_key = f"{model}_msdc"
                    if show_legend and label_key not in added_labels:
                        label = f"{model} MSDC"
                        added_labels.add(label_key)
                    else:
                        label = None

                    for block_idx, value in data_dict.items():
                        ax.bar(
                            block_idx + offset,
                            value,
                            bar_width,
                            color=sdc_colors[model]["msdc"],
                            label=label if block_idx == min(data_dict.keys()) else None,
                            edgecolor="white",
                            linewidth=0.8,
                            alpha=0.95,
                            zorder=2,
                        )

                bar_index += 1

        # Formatting
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Block Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("SDC Value (%)", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFBFC")

        if y_limits:
            y_min, y_max = y_limits
            ax.set_ylim(y_min, y_max)
            # Smart tick spacing for both large and small values
            if y_max < 0.01:
                # For very very small values (< 0.01), use scientific notation
                tick_spacing = y_max / 8
                ticks = np.arange(0, y_max + tick_spacing, tick_spacing)
                ax.set_yticks(ticks)
                ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
            elif y_max < 0.1:
                # For very small values (0.01-0.1), use 3-4 decimal places
                tick_spacing = y_max / 8
                ticks = np.arange(0, y_max + tick_spacing, tick_spacing)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.4f}"))
            elif y_max < 1:
                # For small values (0.1-1), use 2 decimal places
                tick_spacing = y_max / 8
                ticks = np.arange(0, y_max + tick_spacing, tick_spacing)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
            elif y_max < 10:
                # For small values (1-10), use 1 decimal place
                tick_spacing = max(0.5, y_max / 8)
                ax.set_yticks(np.arange(0, y_max + tick_spacing, tick_spacing))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
            else:
                # For larger values, use integer spacing
                tick_spacing = max(1, int(y_max / 8))
                ax.set_yticks(np.arange(0, y_max + tick_spacing, tick_spacing))

        ax.set_xticks(blocks)
        ax.set_xticklabels(blocks, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#95A5A6")
        ax.spines["bottom"].set_color("#95A5A6")

        if show_legend and added_labels:
            legend = ax.legend(
                loc="best",
                fontsize=9,
                framealpha=0.95,
                edgecolor="#BDC3C7",
                fancybox=True,
                shadow=True,
            )
            legend.get_frame().set_facecolor("#FFFFFF")


def main() -> None:
    data = create_empty_data()

    for model, path in FILES.items():
        runs = load_json(path)
        fill_data(runs, model, data)

    # Create interactive plot
    plotter = InteractivePlot(data)
    plt.show()


if __name__ == "__main__":
    main()
