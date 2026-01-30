#!/usr/bin/env python3
"""
Interactive plot tool for ViT activation analysis results.

Creates two types of graphs inspired by Lucas Roquet et al. 2026:
1. Activation Ranges (Fig 2 style): Min/max activation per layer for Block, MHA, MLP
2. Activation Distributions (Fig 3 style): Histogram of activation values

Usage:
    python plot_activations.py activations_vit_tiny.json activations_deit_tiny.json
    python plot_activations.py new_runs/activations_*.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons, Slider, RangeSlider

# Color palette for components (consistent with paper)
COMPONENT_COLORS = {
    "block": "#3498DB",  # Blue
    "mha": "#E67E22",    # Orange
    "mlp": "#27AE60",    # Green
}

# Multiple model color variations
MODEL_PALETTES = [
    {"block": "#3498DB", "mha": "#E67E22", "mlp": "#27AE60"},  # Default
    {"block": "#9B59B6", "mha": "#E74C3C", "mlp": "#1ABC9C"},  # Purple/Red/Teal
    {"block": "#2980B9", "mha": "#D35400", "mlp": "#229954"},  # Darker
    {"block": "#5DADE2", "mha": "#F39C12", "mlp": "#58D68D"},  # Lighter
]


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_label_from_path(path: str) -> str:
    """Extract a clean label from file path."""
    name = Path(path).stem
    for prefix in ["activations_", "activation_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    if len(name) > 25:
        name = name[:22] + "..."
    return name


class CombinedActivationPlot:
    """
    Combined interactive plot with both Fig 2 (ranges) and Fig 3 (distributions) views.
    Supports both detailed (per-layer) and combined (paper style, per-block) views.
    """

    def __init__(self, data: dict, models: list[str], colors: dict):
        self.data = data
        self.models = models
        self.colors = colors
        self.show_models = {m: True for m in models}
        self.show_block = True
        self.show_mha = True
        self.show_mlp = True
        self.show_combined_all = False  # Show combined distribution of all components
        self.use_log_scale = True
        self.show_legend = True
        self.view_mode = "ranges"  # "ranges" or "distributions"
        self.range_style = "detailed"  # "detailed" or "combined"

        # Zoom state for distributions
        self.dist_x_range = None

        self.setup_figure()

    def setup_figure(self):
        self.fig = plt.figure(figsize=(18, 11), facecolor="#F8F9FA")

        # Main plot area
        self.main_area = plt.subplot2grid((26, 28), (0, 0), colspan=20, rowspan=22)
        self.main_area.axis("off")

        # Zoom sliders area (for distributions)
        self.slider_ax_x = plt.subplot2grid((26, 28), (23, 0), colspan=20, rowspan=1)
        self.slider_ax_y = plt.subplot2grid((26, 28), (24, 0), colspan=20, rowspan=1)
        self.slider_ax_x.set_visible(False)
        self.slider_ax_y.set_visible(False)

        # Control panels
        self.view_ax = plt.subplot2grid((26, 28), (0, 21), colspan=7, rowspan=3)
        self.style_ax = plt.subplot2grid((26, 28), (3, 21), colspan=7, rowspan=3)
        self.model_ax = plt.subplot2grid((26, 28), (7, 21), colspan=7, rowspan=min(len(self.models) + 1, 5))
        self.component_ax = plt.subplot2grid((26, 28), (13, 21), colspan=7, rowspan=5)
        self.options_ax = plt.subplot2grid((26, 28), (18, 21), colspan=7, rowspan=3)

        self.setup_controls()
        self.update_plot()

    def setup_controls(self):
        # View mode selection
        self.view_ax.set_title("View Mode", fontsize=10, fontweight="bold")
        self.view_radio = RadioButtons(self.view_ax, ("Activation Ranges", "Distributions"), active=0)
        self.view_radio.on_clicked(self.on_view_change)

        # Range style selection (detailed vs combined)
        self.style_ax.set_title("Range Style", fontsize=10, fontweight="bold")
        self.style_radio = RadioButtons(self.style_ax, ("Detailed (Per-Layer)", "Combined (Paper)"), active=0)
        self.style_radio.on_clicked(self.on_style_change)

        # Model selection
        self.model_ax.set_title("Models", fontsize=10, fontweight="bold")
        self.model_check = CheckButtons(self.model_ax, self.models, [True] * len(self.models))
        self.model_check.on_clicked(self.on_model_change)

        # Component selection
        self.component_ax.set_title("Components", fontsize=10, fontweight="bold")
        self.component_check = CheckButtons(
            self.component_ax, ["Block", "MHA", "MLP", "Combined All"], [True, True, True, False]
        )
        self.component_check.on_clicked(self.on_component_change)

        # Options
        self.options_ax.set_title("Options", fontsize=10, fontweight="bold")
        self.options_check = CheckButtons(self.options_ax, ["Log Scale", "Legend"], [True, True])
        self.options_check.on_clicked(self.on_options_change)

    def on_view_change(self, label):
        self.view_mode = "ranges" if label == "Activation Ranges" else "distributions"
        self.update_plot()

    def on_style_change(self, label):
        self.range_style = "detailed" if "Detailed" in label else "combined"
        self.update_plot()

    def on_model_change(self, label):
        self.show_models[label] = not self.show_models[label]
        self.update_plot()

    def on_component_change(self, label):
        if label == "Block":
            self.show_block = not self.show_block
        elif label == "MHA":
            self.show_mha = not self.show_mha
        elif label == "MLP":
            self.show_mlp = not self.show_mlp
        elif label == "Combined All":
            self.show_combined_all = not self.show_combined_all
        self.update_plot()

    def on_options_change(self, label):
        if label == "Log Scale":
            self.use_log_scale = not self.use_log_scale
        elif label == "Legend":
            self.show_legend = not self.show_legend
        self.update_plot()

    def update_plot(self):
        # Clear previous axes
        for ax in self.fig.axes[:]:
            if ax not in [self.view_ax, self.style_ax, self.model_ax, self.component_ax,
                          self.options_ax, self.slider_ax_x, self.slider_ax_y]:
                if ax != self.main_area:
                    self.fig.delaxes(ax)

        self.main_area.clear()
        self.main_area.axis("off")

        # Show/hide sliders and style selector based on view mode
        if self.view_mode == "distributions":
            self.slider_ax_x.set_visible(True)
            self.slider_ax_y.set_visible(True)
            self.style_ax.set_visible(False)
        else:
            self.slider_ax_x.set_visible(False)
            self.slider_ax_y.set_visible(False)
            self.style_ax.set_visible(True)

        if self.view_mode == "ranges":
            if self.range_style == "detailed":
                self.plot_ranges_detailed()
            else:
                self.plot_ranges_combined()
        else:
            self.plot_distributions()

        self.fig.canvas.draw_idle()

    def plot_ranges_detailed(self):
        """Plot activation ranges with all individual layers (detailed view)."""
        gs = self.main_area.get_subplotspec().subgridspec(1, 1)
        ax = self.fig.add_subplot(gs[0, 0])

        active_models = [m for m in self.models if self.show_models[m]]
        if not active_models:
            ax.text(0.5, 0.5, "No models selected", ha="center", va="center", fontsize=14)
            return

        components = []
        if self.show_mha:
            components.append("mha")
        if self.show_mlp:
            components.append("mlp")
        if self.show_block:
            components.append("block")

        if not components:
            ax.text(0.5, 0.5, "No components selected", ha="center", va="center", fontsize=14)
            return

        added_labels = set()
        all_x_values = []

        for model_idx, model in enumerate(active_models):
            model_data = self.data[model]
            colors = self.colors.get(model, MODEL_PALETTES[model_idx % len(MODEL_PALETTES)])

            # Get layers data
            if "layers" in model_data:
                layers_data = model_data["layers"]
            else:
                layers_data = {}
                for comp in ["block", "mha", "mlp"]:
                    comp_layers = model_data.get("ranges", {}).get(comp, {}).get("layers", {})
                    for idx, info in comp_layers.items():
                        layers_data[idx] = {**info, "component": comp}

            if not layers_data:
                continue

            for comp in components:
                comp_layers = [(int(idx), info) for idx, info in layers_data.items()
                              if info.get("component") == comp]

                if not comp_layers:
                    continue

                comp_layers.sort(key=lambda x: x[0])
                color = colors[comp]

                label_key = f"{model}_{comp}"
                if len(active_models) == 1:
                    label = comp.upper() if self.show_legend and label_key not in added_labels else None
                else:
                    label = f"{model} {comp.upper()}" if self.show_legend and label_key not in added_labels else None

                if label:
                    added_labels.add(label_key)

                for i, (layer_idx, info) in enumerate(comp_layers):
                    min_v = info["min"]
                    max_v = info["max"]
                    all_x_values.append(layer_idx)

                    ax.vlines(layer_idx, min_v, max_v, colors=color, linewidth=1.5, alpha=0.7)
                    ax.scatter([layer_idx], [min_v], color=color, marker="_", s=40, zorder=3, linewidths=1.5)
                    ax.scatter([layer_idx], [max_v], color=color, marker="o", s=20,
                              label=label if i == 0 else None, zorder=3)

        if not all_x_values:
            ax.text(0.5, 0.5, "No layer data available", ha="center", va="center", fontsize=14)
            return

        self._format_range_axis(ax, all_x_values, "Detailed View - All Layers")

    def plot_ranges_combined(self):
        """Plot activation ranges with combined per-block view (paper style).

        X-axis shows actual layer indices where each component ends.
        For Swin: stage wrappers (block_idx % 100 >= 98) only show Block envelope.
        """
        gs = self.main_area.get_subplotspec().subgridspec(1, 1)
        ax = self.fig.add_subplot(gs[0, 0])

        active_models = [m for m in self.models if self.show_models[m]]
        if not active_models:
            ax.text(0.5, 0.5, "No models selected", ha="center", va="center", fontsize=14)
            return

        components = []
        if self.show_mha:
            components.append("mha")
        if self.show_mlp:
            components.append("mlp")
        if self.show_block:
            components.append("block")

        if not components:
            ax.text(0.5, 0.5, "No components selected", ha="center", va="center", fontsize=14)
            return

        added_labels = set()
        all_x_values = []
        x_tick_positions = []
        x_tick_labels = []

        for model_idx, model in enumerate(active_models):
            model_data = self.data[model]
            colors = self.colors.get(model, MODEL_PALETTES[model_idx % len(MODEL_PALETTES)])

            # Get block-aggregated data
            block_agg = model_data.get("block_aggregated", {})

            if not block_agg:
                # Fallback: compute from layers data
                block_agg = self._compute_block_aggregated(model_data)

            if not block_agg:
                continue

            # Get sorted block indices
            block_indices = sorted([int(b) for b in block_agg.keys()])

            # Detect if this is Swin-style (has compound indices like 100, 200, etc.)
            is_swin_style = any(idx >= 100 for idx in block_indices)

            # For combined view, iterate through blocks in order and plot MHA, MLP, Block
            for block_idx in block_indices:
                block_data = block_agg[str(block_idx)]

                # Check if this is a stage wrapper (Swin: indices ending in 98 or 99)
                is_stage_wrapper = is_swin_style and (block_idx % 100 >= 98)

                # Determine which components to plot for this block
                if is_stage_wrapper:
                    # Stage wrappers only show Block (the envelope)
                    comps_to_plot = ["block"] if "block" in components else []
                else:
                    # Normal blocks show MHA, MLP, Block
                    comps_to_plot = [c for c in ["mha", "mlp", "block"] if c in components]

                for comp in comps_to_plot:
                    if comp not in block_data:
                        continue

                    comp_data = block_data[comp]
                    min_v = comp_data.get("min", float("inf"))
                    max_v = comp_data.get("max", float("-inf"))

                    if min_v == float("inf") or max_v == float("-inf"):
                        continue

                    # Use actual layer index - must be present
                    layer_idx = comp_data.get("last_layer_idx", -1)
                    if layer_idx < 0:
                        continue  # Skip if no valid layer index

                    color = colors[comp]
                    label_key = f"{model}_{comp}"

                    if len(active_models) == 1:
                        label = comp.upper() if self.show_legend and label_key not in added_labels else None
                    else:
                        label = f"{model} {comp.upper()}" if self.show_legend and label_key not in added_labels else None

                    all_x_values.append(layer_idx)

                    # Draw vertical line and markers
                    ax.vlines(layer_idx, min_v, max_v, colors=color, linewidth=2, alpha=0.8)
                    ax.scatter([layer_idx], [min_v], color=color, marker="_", s=60, zorder=3, linewidths=2)
                    ax.scatter([layer_idx], [max_v], color=color, marker="o", s=30,
                              label=label, zorder=3)

                    if label:
                        added_labels.add(label_key)

        if not all_x_values:
            ax.text(0.5, 0.5, "No block data available", ha="center", va="center", fontsize=14)
            return

        # Set x-axis ticks to show layer indices
        if x_tick_positions:
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels, fontsize=9)

        num_blocks = len(block_indices) if 'block_indices' in dir() else 12
        self._format_range_axis(ax, all_x_values, f"Combined View - Per Block ({num_blocks} blocks)",
                               xlabel="Layer index")

    def _compute_block_aggregated(self, model_data: dict) -> dict:
        """Compute block-aggregated data from layers if not available.

        Each component (mha, mlp, block) only aggregates its own classified layers.
        """
        block_agg = {}
        layers = model_data.get("layers", {})

        for idx, layer_info in layers.items():
            block_idx = layer_info.get("block_idx")
            if block_idx is None:
                continue

            comp = layer_info.get("component")
            if comp not in ["mha", "mlp", "block"]:
                continue

            block_key = str(block_idx)
            if block_key not in block_agg:
                block_agg[block_key] = {
                    "mha": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
                    "mlp": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
                    "block": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
                }

            layer_idx = int(idx)
            min_v = layer_info["min"]
            max_v = layer_info["max"]

            # Update only the component's own stats
            block_agg[block_key][comp]["min"] = min(block_agg[block_key][comp]["min"], min_v)
            block_agg[block_key][comp]["max"] = max(block_agg[block_key][comp]["max"], max_v)
            block_agg[block_key][comp]["last_layer_idx"] = max(
                block_agg[block_key][comp]["last_layer_idx"], layer_idx
            )

        # Clean up infinities
        for block_key in block_agg:
            for comp in ["mha", "mlp", "block"]:
                if comp in block_agg[block_key] and block_agg[block_key][comp]["min"] == float("inf"):
                    del block_agg[block_key][comp]

        return block_agg

    def _format_range_axis(self, ax, all_x_values, title, xlabel="Layer Index"):
        """Common formatting for range plots with paper-style symmetric log scale."""
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel("Activation Range", fontsize=12, fontweight="bold")

        total_points = len(set(all_x_values))
        ax.set_title(f"{title} ({total_points} points)", fontsize=13, fontweight="bold")

        if self.use_log_scale:
            # Use symmetric log scale like the paper
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)

            # Set y-axis ticks to match paper: -10^3, -10^2, -10^1, 0, 10^1, 10^2, 10^3
            major_ticks = [-1000, -100, -10, 0, 10, 100, 1000]
            minor_ticks = []
            # Add minor ticks between major ticks
            for base in [1, 10, 100]:
                for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                    minor_ticks.extend([base * mult, -base * mult])
            minor_ticks = sorted(set(minor_ticks))

            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)

            # Format y-tick labels like paper: -10^3, -10^2, etc.
            def format_tick(val):
                if val == 0:
                    return "0"
                elif val > 0:
                    if val >= 1000:
                        return f"$10^{{{int(np.log10(val))}}}$"
                    elif val >= 100:
                        return f"$10^2$"
                    elif val >= 10:
                        return f"$10^1$"
                    else:
                        return f"$10^0$"
                else:
                    if val <= -1000:
                        return f"$-10^{{{int(np.log10(-val))}}}$"
                    elif val <= -100:
                        return f"$-10^2$"
                    elif val <= -10:
                        return f"$-10^1$"
                    else:
                        return f"$-10^0$"

            ax.set_yticklabels([format_tick(v) for v in major_ticks])

            # Set y-axis limits to show the full range
            ax.set_ylim(-2000, 2000)

        # Draw horizontal line at y=0
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.7)

        # Smart x-axis ticks for detailed view only
        if xlabel == "Layer Index" and "Layer index" not in xlabel:
            min_x, max_x = min(all_x_values), max(all_x_values)
            num_ticks = min(20, max_x - min_x + 1)
            tick_step = max(1, (max_x - min_x) // num_ticks)
            ax.set_xticks(range(min_x, max_x + 1, tick_step))

        # Grid styling - more visible horizontal lines like the paper
        ax.grid(True, which='major', axis='y', alpha=0.6, linestyle='-', linewidth=0.8, color='#cccccc')
        ax.grid(True, which='minor', axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='#dddddd')
        ax.grid(True, which='major', axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

        ax.set_facecolor("#FAFBFC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if self.show_legend:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=3)

    def plot_distributions(self):
        """Plot activation distributions (Fig 3 style) with zoom sliders."""
        components = []
        if self.show_combined_all:
            components.append(("combined_all", "COMBINED (All Components)"))
        if self.show_block:
            components.append(("block", "BLOCK"))
        if self.show_mha:
            components.append(("mha", "MHA"))
        if self.show_mlp:
            components.append(("mlp", "MLP"))

        if not components:
            gs = self.main_area.get_subplotspec().subgridspec(1, 1)
            ax = self.fig.add_subplot(gs[0, 0])
            ax.text(0.5, 0.5, "No components selected", ha="center", va="center", fontsize=14)
            return

        gs = self.main_area.get_subplotspec().subgridspec(len(components), 1, hspace=0.35)
        active_models = [m for m in self.models if self.show_models[m]]

        # Calculate global x range (include all components for range calculation)
        all_x_min, all_x_max = float("inf"), float("-inf")
        for model in active_models:
            for comp in ["block", "mha", "mlp"]:
                dist_data = self.data[model].get("distributions", {}).get(comp, {})
                if dist_data.get("bin_centers"):
                    all_x_min = min(all_x_min, min(dist_data["bin_centers"]))
                    all_x_max = max(all_x_max, max(dist_data["bin_centers"]))

        if all_x_min == float("inf"):
            all_x_min, all_x_max = -100, 100

        # Setup sliders
        self.slider_ax_x.clear()
        self.slider_ax_y.clear()

        if self.dist_x_range is None:
            self.dist_x_range = (all_x_min, all_x_max)

        self.x_slider = RangeSlider(
            self.slider_ax_x, "X Range",
            all_x_min, all_x_max,
            valinit=self.dist_x_range,
            valstep=(all_x_max - all_x_min) / 100
        )
        self.x_slider.on_changed(self.on_x_slider_change)

        self.y_slider = Slider(
            self.slider_ax_y, "Y Zoom",
            0.1, 10.0,
            valinit=1.0,
            valstep=0.1
        )
        self.y_slider.on_changed(self.on_y_slider_change)

        self.dist_axes = []
        for i, (comp, title) in enumerate(components):
            ax = self.fig.add_subplot(gs[i, 0])
            self.dist_axes.append(ax)

            if not active_models:
                ax.text(0.5, 0.5, "No models selected", ha="center", va="center", fontsize=12)
                continue

            for model_idx, model in enumerate(active_models):
                if comp == "combined_all":
                    # Combine all component distributions for this model
                    combined_centers, combined_counts = self._combine_distributions(model)
                    if combined_centers is None:
                        continue
                    bin_centers = combined_centers
                    counts = combined_counts
                    # Use a distinct color for combined view
                    color = "#8E44AD"  # Purple for combined
                else:
                    dist_data = self.data[model].get("distributions", {}).get(comp, {})
                    bin_centers = dist_data.get("bin_centers", [])
                    counts = dist_data.get("counts", [])

                    if not bin_centers or not counts:
                        continue

                    color = self.colors.get(model, MODEL_PALETTES[model_idx % len(MODEL_PALETTES)])[comp]
                    bin_centers = np.array(bin_centers)
                    counts = np.array(counts)

                label = model if self.show_legend else None
                ax.fill_between(bin_centers, counts, alpha=0.4, color=color, step="mid", label=label)
                ax.step(bin_centers, counts, where="mid", color=color, linewidth=1.5, alpha=0.8)

            ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
            ax.set_xlabel("Activation Value", fontsize=10)
            ax.set_ylabel("Frequency (log)" if self.use_log_scale else "Frequency", fontsize=10)

            if self.use_log_scale:
                ax.set_yscale("log")

            ax.set_xlim(self.dist_x_range)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_facecolor("#FAFBFC")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if self.show_legend and active_models and i == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    def _combine_distributions(self, model: str) -> tuple:
        """Combine distributions from all components (input, block, mha, mlp, output) into one.

        Returns (bin_centers, counts) or (None, None) if no data.
        """
        all_bin_centers = []
        all_counts = []
        total_samples = 0

        # Include all components: input, block, mha, mlp, output
        for comp in ["input", "block", "mha", "mlp", "output"]:
            dist_data = self.data[model].get("distributions", {}).get(comp, {})
            bin_centers = dist_data.get("bin_centers", [])
            counts = dist_data.get("counts", [])
            samples = dist_data.get("total_values_seen", 0)

            if bin_centers and counts:
                all_bin_centers.append(np.array(bin_centers))
                all_counts.append(np.array(counts))
                total_samples += samples

        if not all_bin_centers:
            return None, None

        # Find global min/max across all components
        global_min = min(bc.min() for bc in all_bin_centers)
        global_max = max(bc.max() for bc in all_bin_centers)

        # Create unified bins
        num_bins = 1000
        unified_edges = np.linspace(global_min, global_max, num_bins + 1)
        unified_centers = (unified_edges[:-1] + unified_edges[1:]) / 2
        unified_counts = np.zeros(num_bins)

        # Re-bin each component's data into unified bins
        for bin_centers, counts in zip(all_bin_centers, all_counts):
            # Find which unified bin each original bin falls into
            for j, (center, count) in enumerate(zip(bin_centers, counts)):
                # Find the closest unified bin
                bin_idx = np.searchsorted(unified_edges, center) - 1
                bin_idx = max(0, min(bin_idx, num_bins - 1))
                unified_counts[bin_idx] += count

        return unified_centers, unified_counts

    def on_x_slider_change(self, val):
        self.dist_x_range = val
        for ax in getattr(self, 'dist_axes', []):
            if ax in self.fig.axes:
                ax.set_xlim(val)
        self.fig.canvas.draw_idle()

    def on_y_slider_change(self, val):
        for ax in getattr(self, 'dist_axes', []):
            if ax in self.fig.axes:
                ylim = ax.get_ylim()
                if self.use_log_scale:
                    ax.set_ylim(ylim[0], ylim[1] / val)
                else:
                    ax.set_ylim(0, ylim[1] / val)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive plot tool for ViT activation analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_activations.py activations_vit_tiny.json activations_deit_tiny.json
    python plot_activations.py new_runs/activations_*.json

Creates two graph types from the paper:
  - Activation Ranges (Fig 2): Min/max per layer for Block, MHA, MLP
    - Detailed: Shows all individual layers
    - Combined (Paper): Shows aggregated min/max per block (MHA, MLP, Block)
  - Distributions (Fig 3): Histogram of activation values with zoom controls
        """
    )
    parser.add_argument("files", nargs="+", help="Activation JSON result files to plot")

    args = parser.parse_args()

    if not args.files:
        print("Error: No input files specified")
        sys.exit(1)

    for f in args.files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)

    models = []
    colors = {}
    data = {}

    for i, path in enumerate(args.files):
        label = get_label_from_path(path)
        models.append(label)
        colors[label] = MODEL_PALETTES[i % len(MODEL_PALETTES)]
        data[label] = load_json(path)

    print(f"Loaded {len(models)} activation datasets:")
    for label in models:
        stats = data[label].get("statistics", {})
        print(f"  - {label}: {stats.get('total_samples', '?')} samples, "
              f"{stats.get('total_layers', '?')} layers, "
              f"{stats.get('num_blocks', '?')} blocks")

    plotter = CombinedActivationPlot(data, models, colors)
    plt.show()


if __name__ == "__main__":
    main()
