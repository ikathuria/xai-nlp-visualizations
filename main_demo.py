from src.evaluation_metrics import EvaluationMetrics
from src.visualization import AdvancedVisualizer, create_publication_figure
from src.trajectory import EnhancedTrajectoryExtractor
import traceback
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')


class XAIVisualizationPipeline:
    """
    Complete pipeline for XAI visualization of transformer models.
    Integrates trajectory extraction, visualization, and evaluation.
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 device: str = None):
        """
        Initialize the XAI visualization pipeline.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device

        print(f"Initializing XAI Pipeline with model: {model_name}")

        # Initialize components
        try:
            self.extractor = EnhancedTrajectoryExtractor(model_name, device)
            print("✓ Trajectory extractor initialized")
        except Exception as e:
            print(f"✗ Error initializing extractor: {e}")
            raise

        try:
            self.visualizer = AdvancedVisualizer()
            print("✓ Visualizer initialized")
        except Exception as e:
            print(f"✗ Error initializing visualizer: {e}")
            raise

        try:
            self.evaluator = EvaluationMetrics(model_name, device)
            print("✓ Evaluator initialized")
        except Exception as e:
            print(f"✗ Error initializing evaluator: {e}")
            # Make evaluator optional for now
            self.evaluator = None

        # Create output directories
        os.makedirs("figures", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        print("Pipeline initialized successfully!")

    def analyze_single_text(self,
                            text: str,
                            idx: int = 0,
                            methods: List[str] = None,
                            modes: List[str] = None) -> Dict[str, Any]:
        """
        Complete analysis of a single text input with improved error handling.
        """
        if methods is None:
            methods = ['cls_attention',
                       'attention_rollout', 'integrated_gradients']

        if modes is None:
            modes = ['full', 'exclude', 'downweight']

        print(f"\nAnalyzing text: '{text[:50]}...'")

        results = {
            'text': text,
            'model': self.model_name,
            'methods': {},
            'visualizations': {},
            'evaluation': {}
        }

        # Extract using different methods
        for method in methods:
            print(f"  Extracting {method}...")

            try:
                if method == 'cls_attention':
                    tokens, data = self.extractor.extract_layerwise_trajectories(
                        text, 'cls_attention')
                elif method == 'attention_rollout':
                    tokens, data = self.extractor.extract_attention_rollout(
                        text)
                elif method == 'integrated_gradients':
                    tokens, data = self.extractor.extract_gradient_attribution(
                        text, method='integrated_gradients')
                elif method == 'multi_head':
                    tokens, head_data, cls_data = self.extractor.extract_multi_head_attention(
                        text)
                    data = np.mean(cls_data, axis=1)  # Average across heads
                else:
                    print(f"    Unknown method: {method}")
                    continue

                results['methods'][method] = {
                    'tokens': tokens,
                    'data': data
                }
                print(f"    ✓ {method} completed successfully")

            except Exception as e:
                print(f"    ✗ Error with {method}: {e}")
                continue

        # Create visualizations for each mode (only if we have cls_attention)
        if 'cls_attention' in results['methods']:
            base_tokens = results['methods']['cls_attention']['tokens']
            base_data = results['methods']['cls_attention']['data']

            for mode in modes:
                print(f"  Creating {mode} visualization...")

                try:
                    # Create trajectory plot - use matplotlib directly for now
                    plt.figure(figsize=(12, 8))

                    # Apply filtering based on mode
                    display_tokens, display_data = self.visualizer._filter_tokens(
                        base_tokens.copy(), base_data.copy(), mode, 0.1
                    )

                    # Plot trajectories
                    for i, token in enumerate(display_tokens):
                        if i < display_data.shape[1]:  # Ensure index is valid
                            plt.plot(range(1, len(display_data) + 1), display_data[:, i],
                                     label=token, linewidth=2, alpha=0.8)

                    plt.title(f"Token Trajectories - {mode.title()} Mode")
                    plt.xlabel("Layer")
                    plt.ylabel("Importance")

                    if len(display_tokens) <= 10:  # Only show legend if not too crowded
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                    plt.tight_layout()
                    save_path = f"figures/trajectories_{idx}_{mode}.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    results['visualizations'][f'{mode}_trajectory'] = save_path
                    print(f"    ✓ {mode} visualization saved to {save_path}")

                except Exception as e:
                    print(f"    ✗ Error creating {mode} visualization: {e}")
                    continue

        # Multi-method comparison (simplified)
        if len(results['methods']) > 1:
            print("  Creating method comparison...")

            try:
                plt.figure(figsize=(15, 6))

                comparison_data = {}
                comparison_tokens = None

                for method, method_results in results['methods'].items():
                    tokens = method_results['tokens']
                    data = method_results['data']

                    if comparison_tokens is None:
                        comparison_tokens = tokens

                    # Extract single importance value per token
                    if data.ndim > 1:
                        comparison_data[method] = data[-1]  # Use final layer
                    else:
                        comparison_data[method] = data

                # Create subplots for each method
                n_methods = len(comparison_data)
                for i, (method, values) in enumerate(comparison_data.items(), 1):
                    plt.subplot(1, n_methods, i)

                    # Apply exclude mode filtering
                    filtered_tokens, filtered_values = self.visualizer._filter_tokens(
                        comparison_tokens.copy(), values.copy(), 'exclude'
                    )

                    plt.bar(range(len(filtered_tokens)),
                            filtered_values, alpha=0.7)
                    plt.title(f"{method}")
                    plt.xticks(range(len(filtered_tokens)),
                               filtered_tokens, rotation=45, ha='right')

                plt.tight_layout()
                save_path = f"figures/method_comparison_{idx}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                results['visualizations']['method_comparison'] = save_path
                print(f"    ✓ Method comparison saved to {save_path}")

            except Exception as e:
                print(f"    ✗ Error creating method comparison: {e}")

        # Evaluation (simplified and optional)
        if self.evaluator is not None:
            print("  Running evaluation...")
            try:
                # Simple evaluation - just compute basic metrics
                if 'cls_attention' in results['methods']:
                    attention_data = results['methods']['cls_attention']['data']

                    # Compute some basic metrics
                    stability_metrics = {}
                    if attention_data.ndim == 2:
                        # Layer-to-layer correlations
                        correlations = []
                        for i in range(attention_data.shape[0] - 1):
                            from scipy.stats import pearsonr
                            corr, _ = pearsonr(
                                attention_data[i], attention_data[i + 1])
                            if not np.isnan(corr):
                                correlations.append(corr)

                        if correlations:
                            stability_metrics['mean_layer_correlation'] = np.mean(
                                correlations)

                    results['evaluation'] = {'stability': stability_metrics}
                    print("    ✓ Basic evaluation completed")

            except Exception as e:
                print(f"    ✗ Evaluation error: {e}")
                results['evaluation'] = {'error': str(e)}

        print("  Analysis complete!")
        return results

    def create_publication_figures(self,
                                   texts: List[str],
                                   save_prefix: str = "publication") -> List[str]:
        """
        Create publication-ready figures for paper.
        """
        print("\nCreating publication figures...")
        figure_paths = []

        for i, text in enumerate(texts):
            print(f"  Processing example {i+1}: '{text[:30]}...'")

            try:
                # Extract attention trajectories
                tokens, trajectories = self.extractor.extract_layerwise_trajectories(
                    text)

                # Create comprehensive figure showing all modes
                save_path = f"figures/{save_prefix}_example_{i+1}.png"

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                modes = ["full", "exclude", "downweight"]
                titles = ["Full (with special tokens)",
                          "Exclude special tokens", "Downweight special tokens"]

                for j, (mode, title) in enumerate(zip(modes, titles)):
                    filtered_tokens, filtered_traj = self.visualizer._filter_tokens(
                        tokens, trajectories, mode)

                    for k, token in enumerate(filtered_tokens):
                        if k < filtered_traj.shape[1]:
                            axes[j].plot(range(1, len(filtered_traj) + 1), filtered_traj[:, k],
                                         label=token, linewidth=2, alpha=0.8)

                    axes[j].set_title(title)
                    axes[j].set_xlabel("Layer")
                    axes[j].set_ylabel("Importance")

                    if len(filtered_tokens) <= 8:
                        axes[j].legend(bbox_to_anchor=(
                            1.05, 1), loc='upper left')

                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                figure_paths.append(save_path)
                print(f"    ✓ Publication figure saved to {save_path}")

                # Create attention heatmap for final layer
                try:
                    _, head_attention, _ = self.extractor.extract_multi_head_attention(
                        text)
                    # Average across heads
                    final_layer_attention = head_attention[-1].mean(axis=0)

                    heatmap_path = f"figures/{save_prefix}_heatmap_{i+1}.png"

                    plt.figure(figsize=(10, 8))
                    plt.imshow(final_layer_attention,
                               cmap='Blues', aspect='equal')
                    plt.colorbar(label='Attention Weight')
                    plt.title(f"Attention Heatmap - Final Layer")

                    # Set ticks and labels
                    plt.xticks(range(len(tokens)), tokens,
                               rotation=45, ha='right')
                    plt.yticks(range(len(tokens)), tokens)
                    plt.xlabel("Attended To")
                    plt.ylabel("Attending From")

                    plt.tight_layout()
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    figure_paths.append(heatmap_path)
                    print(f"    ✓ Heatmap saved to {heatmap_path}")

                except Exception as e:
                    print(f"    ✗ Error creating heatmap: {e}")

            except Exception as e:
                print(f"    ✗ Error processing example {i+1}: {e}")
                continue

        print(f"  Created {len(figure_paths)} publication figures")
        return figure_paths

    def run_benchmark_study(self,
                            texts: List[str],
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Run simplified benchmark study on multiple texts.
        """
        print(f"\nRunning simplified benchmark study on {len(texts)} texts...")

        all_results = []

        for i, text in enumerate(texts):
            print(f"Evaluating text {i+1}/{len(texts)}: '{text[:30]}...'")

            try:
                # Extract different attribution methods (with error handling)
                methods_data = {}

                try:
                    tokens, attention_traj = self.extractor.extract_layerwise_trajectories(
                        text)
                    methods_data['attention_trajectory'] = attention_traj[-1] if attention_traj.ndim > 1 else attention_traj
                except Exception as e:
                    print(f"  ✗ Error extracting attention trajectory: {e}")

                try:
                    _, rollout = self.extractor.extract_attention_rollout(text)
                    methods_data['attention_rollout'] = rollout
                except Exception as e:
                    print(f"  ✗ Error extracting attention rollout: {e}")

                # Skip integrated gradients for now due to the shape error
                print("  Skipping integrated gradients due to implementation issues")

                # Simple evaluation
                result = {
                    'text': text[:100],
                    'methods': list(methods_data.keys()),
                    'num_tokens': len(tokens) if 'tokens' in locals() else 0
                }

                # Basic stability analysis if we have trajectory data
                if 'attention_trajectory' in methods_data:
                    try:
                        # Compute some basic statistics
                        traj_data = attention_traj if 'attention_traj' in locals() else None
                        if traj_data is not None and traj_data.ndim == 2:
                            token_variances = np.var(traj_data, axis=0)
                            result['stability'] = {
                                'mean_token_variance': np.mean(token_variances),
                                'max_token_variance': np.max(token_variances)
                            }
                    except Exception as e:
                        print(f"  ✗ Error computing stability: {e}")

                all_results.append(result)
                print(f"  ✓ Text {i+1} processed successfully")

            except Exception as e:
                print(f"  ✗ Error evaluating text {i+1}: {e}")
                # Add minimal result to maintain consistency
                all_results.append({
                    'text': text[:100],
                    'error': str(e),
                    'methods': [],
                    'num_tokens': 0
                })
                continue

        # Aggregate results
        aggregated = {
            'num_texts': len(texts),
            'successful_analyses': len([r for r in all_results if 'error' not in r]),
            'individual_results': all_results
        }

        if save_results:
            import json
            try:
                with open('evaluation_results.json', 'w') as f:
                    json.dump(aggregated, f, indent=2, default=str)
                print(f"✓ Results saved to evaluation_results.json")
            except Exception as e:
                print(f"✗ Error saving results: {e}")

        return aggregated

    def generate_demo_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate comprehensive demo report.
        """
        report_lines = []
        report_lines.append("# XAI Visualization Pipeline - Demo Report")
        report_lines.append("=" * 60)

        import datetime
        report_lines.append(f"\n**Model**: {self.model_name}")
        report_lines.append(f"**Number of examples**: {len(results)}")
        report_lines.append(
            f"**Analysis date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for i, result in enumerate(results, 1):
            report_lines.append(f"\n## Example {i}")
            report_lines.append(f"**Text**: {result['text'][:100]}...")

            if 'methods' in result:
                report_lines.append(
                    f"**Methods used**: {', '.join(result['methods'].keys())}")

            if 'visualizations' in result:
                report_lines.append("**Generated figures**:")
                for viz_name, path in result['visualizations'].items():
                    report_lines.append(f"  - {viz_name}: {path}")

        return "\n".join(report_lines)


def main():
    """Main demonstration function with better error handling."""
    print("🚀 XAI Visualization Pipeline Demo")
    print("=" * 50)

    # Example texts for demonstration
    example_texts = [
        "I love this movie! The acting was fantastic and the plot was engaging.",
        "This product is terrible. Poor quality and overpriced.",
        "The research paper presents interesting findings about neural networks.",
        "Climate change poses significant challenges for future generations."
    ]

    try:
        # Initialize pipeline
        pipeline = XAIVisualizationPipeline(
            model_name="distilbert-base-uncased-finetuned-sst-2-english"
        )

        # Analyze each text
        all_results = []
        for i, text in enumerate(example_texts):
            print(f"\n{'='*20} EXAMPLE {i+1} {'='*20}")
            result = pipeline.analyze_single_text(text, idx=i+1)
            all_results.append(result)

        # Create publication figures
        print(f"\n{'='*20} PUBLICATION FIGURES {'='*20}")
        figure_paths = pipeline.create_publication_figures(example_texts[:2])

        # Run benchmark evaluation (simplified)
        print(f"\n{'='*20} BENCHMARK EVALUATION {'='*20}")
        benchmark_results = pipeline.run_benchmark_study(example_texts)

        # Generate final report
        print(f"\n{'='*20} GENERATING REPORT {'='*20}")
        report = pipeline.generate_demo_report(all_results)

        # Save report
        try:
            with open("reports/demo_report.md", "w") as f:
                f.write(report)
            print("✓ Demo report saved to reports/demo_report.md")
        except Exception as e:
            print(f"✗ Error saving report: {e}")

        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Check the 'figures/' directory for visualizations")
        print(f"📋 Check the 'reports/' directory for analysis reports")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
