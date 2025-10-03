import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class AdvancedVisualizer:
    """
    Advanced visualization toolkit for transformer attention and attribution analysis.
    Includes interactive plots, multi-view analysis, and publication-ready figures.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 12)

    def _filter_tokens(self, tokens: List[str], data: np.ndarray,
                       mode: str = "full", downweight_factor: float = 0.1) -> Tuple[List[str], np.ndarray]:
        """Enhanced token filtering with better special token detection."""
        data = data.copy()

        # Detect special tokens more robustly
        special_indices = []
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '<s>', '</s>', '<|endoftext|>', '[PAD]']:
                special_indices.append(i)

        if mode == "exclude" and special_indices:
            # Remove all special tokens
            keep_indices = [i for i in range(
                len(tokens)) if i not in special_indices]
            tokens = [tokens[i] for i in keep_indices]
            if data.ndim == 1:
                data = data[keep_indices]
            else:
                data = data[:, keep_indices]

        elif mode == "downweight" and special_indices:
            # Downweight all special tokens
            for idx in special_indices:
                if data.ndim == 1:
                    data[idx] *= downweight_factor
                else:
                    data[:, idx] *= downweight_factor

        return tokens, data

    def plot_layerwise_trajectories(self,
                                    tokens: List[str],
                                    trajectories: np.ndarray,
                                    title: str = "Token Importance Trajectories",
                                    mode: str = "full",
                                    downweight_factor: float = 0.1,
                                    highlight_tokens: List[str] = None,
                                    save_path: str = None) -> go.Figure:
        """
        Create interactive layerwise trajectory plot with enhanced features.

        Args:
            tokens: List of token strings
            trajectories: Shape (num_layers, seq_len) 
            title: Plot title
            mode: Token filtering mode
            downweight_factor: Factor for downweighting special tokens
            highlight_tokens: Tokens to highlight with special styling
            save_path: Path to save figure
        """
        tokens, trajectories = self._filter_tokens(
            tokens, trajectories, mode, downweight_factor)

        num_layers, seq_len = trajectories.shape

        # Create interactive plot
        fig = go.Figure()

        # Add trajectory for each token
        for i, token in enumerate(tokens):
            # Determine line style
            line_width = 3 if highlight_tokens and token in highlight_tokens else 2
            line_dash = 'solid' if not token.startswith(
                '#') else 'dash'  # Subword tokens dashed

            fig.add_trace(go.Scatter(
                x=list(range(1, num_layers + 1)),
                y=trajectories[:, i],
                mode='lines+markers',
                name=token,
                line=dict(width=line_width, dash=line_dash),
                marker=dict(size=6),
                hovertemplate=f'<b>{token}</b><br>Layer: %{{x}}<br>Importance: %{{y:.3f}}<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Layer",
            yaxis_title="Importance",
            width=1000,
            height=600,
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            fig.write_image(save_path, width=1000, height=600)

        return fig

    def plot_attention_heatmap(self,
                               tokens: List[str],
                               attention_matrix: np.ndarray,
                               title: str = "Attention Heatmap",
                               layer: int = None,
                               head: int = None,
                               save_path: str = None) -> plt.Figure:
        """
        Create publication-ready attention heatmap.

        Args:
            tokens: List of token strings
            attention_matrix: Shape (seq_len, seq_len) or (heads, seq_len, seq_len)
            title: Plot title
            layer: Layer number for title
            head: Head number for title  
            save_path: Path to save figure
        """
        if attention_matrix.ndim == 3:
            # If multi-head, average across heads
            attention_matrix = attention_matrix.mean(axis=0)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create heatmap
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='equal')

        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)

        # Add title with layer/head info
        full_title = title
        if layer is not None:
            full_title += f" (Layer {layer}"
            if head is not None:
                full_title += f", Head {head}"
            full_title += ")"
        ax.set_title(full_title)

        ax.set_xlabel("Attended To")
        ax.set_ylabel("Attending From")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_multi_method_comparison(self,
                                     tokens: List[str],
                                     methods_data: Dict[str, np.ndarray],
                                     title: str = "Multi-Method Comparison",
                                     mode: str = "full",
                                     save_path: str = None) -> go.Figure:
        """
        Compare multiple attribution methods side by side.

        Args:
            tokens: List of token strings
            methods_data: Dict mapping method names to attribution arrays
            title: Plot title
            mode: Token filtering mode
            save_path: Path to save figure
        """
        # Filter tokens for all methods
        filtered_data = {}
        for method, data in methods_data.items():
            _, filtered = self._filter_tokens(tokens, data, mode)
            filtered_data[method] = filtered

        tokens, _ = self._filter_tokens(
            tokens, list(methods_data.values())[0], mode)

        # Create subplot figure
        num_methods = len(methods_data)
        fig = make_subplots(
            rows=1, cols=num_methods,
            subplot_titles=list(methods_data.keys()),
            shared_yaxes=True
        )

        colors = px.colors.qualitative.Set3[:len(tokens)]

        for col, (method, data) in enumerate(filtered_data.items(), 1):
            if data.ndim == 2:  # Multiple layers
                # Show final layer or average
                values = data[-1] if data.shape[0] > 1 else data[0]
            else:
                values = data

            fig.add_trace(
                go.Bar(
                    x=tokens,
                    y=values,
                    name=f"{method}",
                    marker_color=colors[:len(tokens)],
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

        fig.update_layout(
            title_text=title,
            height=600,
            width=300 * num_methods
        )

        fig.update_xaxes(tickangle=45)

        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            fig.write_image(save_path)

        return fig

    def plot_head_importance_matrix(self,
                                    attention_data: np.ndarray,
                                    title: str = "Head Importance Across Layers",
                                    save_path: str = None) -> plt.Figure:
        """
        Visualize attention head importance across layers.

        Args:
            attention_data: Shape (layers, heads, seq_len, seq_len)
            title: Plot title
            save_path: Path to save figure
        """
        num_layers, num_heads = attention_data.shape[:2]

        # Calculate head importance (variance of attention patterns)
        head_importance = np.zeros((num_layers, num_heads))

        for layer in range(num_layers):
            for head in range(num_heads):
                # Use attention entropy as importance measure
                attn_matrix = attention_data[layer, head]
                entropy = -np.sum(attn_matrix *
                                  np.log(attn_matrix + 1e-8), axis=-1)
                head_importance[layer, head] = entropy.mean()

        fig, ax = plt.subplots(figsize=(num_heads, num_layers), dpi=self.dpi)

        im = ax.imshow(head_importance, cmap='viridis', aspect='auto')

        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_layers))
        ax.set_xticklabels([f'H{i}' for i in range(num_heads)])
        ax.set_yticklabels([f'L{i}' for i in range(num_layers)])

        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Layer')
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Entropy', rotation=270, labelpad=15)

        # Add text annotations
        for layer in range(num_layers):
            for head in range(num_heads):
                text = ax.text(head, layer, f'{head_importance[layer, head]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_counterfactual_analysis(self,
                                     original_data: Tuple[List[str], np.ndarray],
                                     counterfactual_data: Tuple[List[str], np.ndarray],
                                     title: str = "Counterfactual Analysis",
                                     mode: str = "exclude",
                                     save_path: str = None) -> plt.Figure:
        """
        Enhanced counterfactual comparison with statistical testing.
        """
        orig_tokens, orig_traj = original_data
        cf_tokens, cf_traj = counterfactual_data

        # Filter tokens
        orig_tokens, orig_traj = self._filter_tokens(
            orig_tokens, orig_traj, mode)
        cf_tokens, cf_traj = self._filter_tokens(cf_tokens, cf_traj, mode)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

        # Original trajectory
        axes[0, 0].set_title("Original")
        for i, token in enumerate(orig_tokens):
            axes[0, 0].plot(range(1, len(orig_traj) + 1), orig_traj[:, i],
                            label=token, alpha=0.8, linewidth=2)
        axes[0, 0].set_xlabel("Layer")
        axes[0, 0].set_ylabel("Importance")
        if len(orig_tokens) <= 10:
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Counterfactual trajectory
        axes[0, 1].set_title("Counterfactual")
        for i, token in enumerate(cf_tokens):
            axes[0, 1].plot(range(1, len(cf_traj) + 1), cf_traj[:, i],
                            label=token, alpha=0.8, linewidth=2)
        axes[0, 1].set_xlabel("Layer")
        axes[0, 1].set_ylabel("Importance")
        if len(cf_tokens) <= 10:
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Difference heatmap (if same shape)
        if orig_traj.shape == cf_traj.shape:
            diff = orig_traj - cf_traj
            im = axes[1, 0].imshow(diff.T, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_title("Difference (Orig - CF)")
            axes[1, 0].set_xlabel("Layer")
            axes[1, 0].set_ylabel("Token")
            axes[1, 0].set_yticks(range(len(orig_tokens)))
            axes[1, 0].set_yticklabels(orig_tokens)
            plt.colorbar(im, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, "Different sequence lengths\nCannot compute difference",
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Difference (Not Available)")

        # Statistical significance (if same shape)
        if orig_traj.shape == cf_traj.shape:
            # Compute layer-wise differences
            layer_diffs = np.mean(np.abs(diff), axis=1)
            axes[1, 1].bar(range(1, len(layer_diffs) + 1), layer_diffs,
                           alpha=0.7, color='skyblue')
            axes[1, 1].set_title("Mean Absolute Difference by Layer")
            axes[1, 1].set_xlabel("Layer")
            axes[1, 1].set_ylabel("Mean |Difference|")
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def create_summary_report(self,
                              results: Dict[str, any],
                              save_path: str = None) -> str:
        """
        Generate a comprehensive analysis report.

        Args:
            results: Dictionary containing analysis results
            save_path: Path to save report

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("# XAI Visualization Analysis Report")
        report_lines.append("=" * 50)

        if 'model_info' in results:
            report_lines.append(f"\nModel: {results['model_info']['name']}")
            report_lines.append(f"Layers: {results['model_info']['layers']}")
            report_lines.append(f"Heads: {results['model_info']['heads']}")

        if 'special_token_analysis' in results:
            sta = results['special_token_analysis']
            report_lines.append("\n## Special Token Analysis")
            report_lines.append(
                f"Special tokens detected: {sta['special_tokens']}")
            report_lines.append(
                f"Average attention to [CLS]: {sta['cls_attention']:.3f}")
            report_lines.append(
                f"Average attention to [SEP]: {sta['sep_attention']:.3f}")

        if 'method_comparison' in results:
            report_lines.append("\n## Method Comparison")
            for method, score in results['method_comparison'].items():
                report_lines.append(f"{method}: {score:.3f}")

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report


def create_publication_figure(tokens: List[str],
                              trajectories: np.ndarray,
                              mode: str = "exclude",
                              save_path: str = None) -> plt.Figure:
    """
    Create a publication-ready figure showing all three visualization modes.
    """
    visualizer = AdvancedVisualizer()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    modes = ["full", "exclude", "downweight"]
    titles = ["Full (with special tokens)",
              "Exclude special tokens", "Downweight special tokens"]

    for i, (mode, title) in enumerate(zip(modes, titles)):
        filtered_tokens, filtered_traj = visualizer._filter_tokens(
            tokens, trajectories, mode)

        for j, token in enumerate(filtered_tokens):
            axes[i].plot(range(1, len(filtered_traj) + 1), filtered_traj[:, j],
                         label=token, linewidth=2, alpha=0.8)

        axes[i].set_title(title)
        axes[i].set_xlabel("Layer")
        axes[i].set_ylabel("Importance")

        if len(filtered_tokens) <= 8:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
