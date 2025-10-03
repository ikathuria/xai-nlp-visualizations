import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EvaluationMetrics:
    """
    Evaluation metrics for attention and attribution visualization methods.
    Implements faithfulness, plausibility, and other XAI evaluation metrics.
    """

    def __init__(self, model_name: str, device: str = None):
        """
        Initialize evaluation metrics.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device (cuda/cpu)
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name).to(self.device)
            self.model.eval()
            print("✓ Evaluation model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading evaluation model: {e}")
            self.tokenizer = None
            self.model = None

    def faithfulness_correlation(self,
                                 text: str,
                                 attributions: np.ndarray,
                                 perturbation_steps: int = 5) -> float:
        """
        Measure faithfulness by correlating attribution scores with prediction changes
        when tokens are masked/removed. Simplified version to avoid complex errors.
        """
        if self.model is None or self.tokenizer is None:
            return 0.0

        try:
            original_logits = self._get_prediction_logits(text)
            if original_logits is None:
                return 0.0

            original_pred = torch.softmax(original_logits, dim=-1).max().item()

            # Simplified perturbation - just mask a few top tokens
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) != len(attributions):
                print(
                    f"Warning: Token length mismatch: {len(tokens)} vs {len(attributions)}")
                return 0.0

            # Sort tokens by attribution score (descending)
            token_importance = list(
                zip(range(len(attributions)), attributions))
            token_importance.sort(key=lambda x: x[1], reverse=True)

            prediction_drops = []
            cumulative_attributions = []

            masked_indices = set()

            for step in range(min(perturbation_steps, len(tokens))):
                token_idx, attribution = token_importance[step]
                masked_indices.add(token_idx)

                # Create masked text
                masked_text = self._mask_tokens(text, list(masked_indices))
                if masked_text is None:
                    continue

                # Get prediction on masked text
                masked_logits = self._get_prediction_logits(masked_text)
                if masked_logits is None:
                    continue

                masked_pred = torch.softmax(masked_logits, dim=-1).max().item()

                # Calculate prediction drop
                pred_drop = original_pred - masked_pred
                prediction_drops.append(pred_drop)

                # Calculate cumulative attribution
                cumulative_attr = sum(attributions[idx]
                                      for idx in masked_indices)
                cumulative_attributions.append(cumulative_attr)

            # Calculate correlation
            if len(prediction_drops) > 1:
                correlation, _ = pearsonr(
                    cumulative_attributions, prediction_drops)
                return correlation if not np.isnan(correlation) else 0.0
            else:
                return 0.0

        except Exception as e:
            print(f"Error in faithfulness computation: {e}")
            return 0.0

    def plausibility_human_agreement(self,
                                     attributions: np.ndarray,
                                     human_annotations: np.ndarray) -> Dict[str, float]:
        """
        Measure plausibility by comparing with human annotations.
        """
        try:
            # Ensure same length
            min_len = min(len(attributions), len(human_annotations))
            attr_subset = attributions[:min_len]
            human_subset = human_annotations[:min_len]

            # Calculate correlations
            pearson_r, pearson_p = pearsonr(attr_subset, human_subset)
            spearman_r, spearman_p = spearmanr(attr_subset, human_subset)

            return {
                'pearson_correlation': pearson_r if not np.isnan(pearson_r) else 0.0,
                'pearson_p_value': pearson_p if not np.isnan(pearson_p) else 1.0,
                'spearman_correlation': spearman_r if not np.isnan(spearman_r) else 0.0,
                'spearman_p_value': spearman_p if not np.isnan(spearman_p) else 1.0
            }
        except Exception as e:
            print(f"Error in plausibility computation: {e}")
            return {
                'pearson_correlation': 0.0,
                'pearson_p_value': 1.0,
                'spearman_correlation': 0.0,
                'spearman_p_value': 1.0
            }

    def attention_concentration(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """
        Measure attention concentration using entropy and Gini coefficient.
        """
        try:
            if attention_matrix.ndim == 3:
                # Average across layers
                attention_matrix = attention_matrix.mean(axis=0)

            metrics = {}

            # Calculate entropy for each attending position
            entropies = []
            for i in range(attention_matrix.shape[0]):
                attn_dist = attention_matrix[i]
                attn_dist = attn_dist / (attn_dist.sum() + 1e-8)  # Normalize
                entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-8))
                entropies.append(entropy)

            metrics['mean_entropy'] = np.mean(entropies)
            metrics['entropy_std'] = np.std(entropies)

            # Gini coefficient for attention distribution
            def gini_coefficient(x):
                """Calculate Gini coefficient for array x."""
                x = np.sort(x)
                n = len(x)
                if n == 0:
                    return 0.0
                cumsum = np.cumsum(x)
                if cumsum[-1] == 0:
                    return 0.0
                return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

            # Calculate Gini for attention received by each token
            attention_received = attention_matrix.sum(axis=0)
            metrics['gini_coefficient'] = gini_coefficient(attention_received)

            # Special token dominance (if applicable)
            if len(attention_received) > 2:
                # CLS + SEP
                special_token_attention = attention_received[0] + \
                    attention_received[-1]
                total_attention = attention_received.sum()
                metrics['special_token_dominance'] = special_token_attention / \
                    (total_attention + 1e-8)
            else:
                metrics['special_token_dominance'] = 0.0

            return metrics

        except Exception as e:
            print(f"Error in attention concentration computation: {e}")
            return {
                'mean_entropy': 0.0,
                'entropy_std': 0.0,
                'gini_coefficient': 0.0,
                'special_token_dominance': 0.0
            }

    def stability_across_layers(self, trajectories: np.ndarray) -> Dict[str, float]:
        """
        Measure stability of token importance across layers.
        """
        try:
            metrics = {}

            if trajectories.ndim != 2:
                return {'error': 'Invalid trajectory dimensions'}

            # Calculate variance for each token across layers
            token_variances = np.var(trajectories, axis=0)
            metrics['mean_token_variance'] = np.mean(token_variances)
            metrics['max_token_variance'] = np.max(token_variances)

            # Calculate layer-to-layer correlations
            layer_correlations = []
            for i in range(trajectories.shape[0] - 1):
                corr, _ = pearsonr(trajectories[i], trajectories[i + 1])
                if not np.isnan(corr):
                    layer_correlations.append(corr)

            metrics['mean_layer_correlation'] = np.mean(
                layer_correlations) if layer_correlations else 0
            metrics['min_layer_correlation'] = np.min(
                layer_correlations) if layer_correlations else 0

            # Trend consistency (whether tokens consistently increase/decrease)
            trend_consistency = []
            for token_idx in range(trajectories.shape[1]):
                token_traj = trajectories[:, token_idx]
                # Calculate correlation with linear trend
                trend_corr, _ = pearsonr(token_traj, range(len(token_traj)))
                trend_consistency.append(
                    abs(trend_corr) if not np.isnan(trend_corr) else 0)

            metrics['mean_trend_consistency'] = np.mean(trend_consistency)

            return metrics

        except Exception as e:
            print(f"Error in stability computation: {e}")
            return {
                'mean_token_variance': 0.0,
                'max_token_variance': 0.0,
                'mean_layer_correlation': 0.0,
                'min_layer_correlation': 0.0,
                'mean_trend_consistency': 0.0
            }

    def method_agreement(self,
                         method1_scores: np.ndarray,
                         method2_scores: np.ndarray,
                         method_names: Tuple[str, str] = ("Method1", "Method2")) -> Dict[str, float]:
        """
        Measure agreement between two attribution methods.
        """
        try:
            # Ensure same length
            min_len = min(len(method1_scores), len(method2_scores))
            scores1 = method1_scores[:min_len]
            scores2 = method2_scores[:min_len]

            # Normalize scores
            if scores1.max() > scores1.min():
                scores1 = (scores1 - scores1.min()) / \
                    (scores1.max() - scores1.min())
            if scores2.max() > scores2.min():
                scores2 = (scores2 - scores2.min()) / \
                    (scores2.max() - scores2.min())

            # Calculate correlations
            pearson_r, pearson_p = pearsonr(scores1, scores2)
            spearman_r, spearman_p = spearmanr(scores1, scores2)

            # Top-k agreement
            k_values = [1, 3, 5]
            top_k_agreements = {}

            for k in k_values:
                if k <= min_len:
                    top_k1 = set(np.argsort(scores1)[-k:])
                    top_k2 = set(np.argsort(scores2)[-k:])
                    agreement = len(top_k1.intersection(top_k2)) / k
                    top_k_agreements[f'top_{k}_agreement'] = agreement

            return {
                'pearson_correlation': pearson_r if not np.isnan(pearson_r) else 0.0,
                'spearman_correlation': spearman_r if not np.isnan(spearman_r) else 0.0,
                **top_k_agreements
            }

        except Exception as e:
            print(f"Error in method agreement computation: {e}")
            return {
                'pearson_correlation': 0.0,
                'spearman_correlation': 0.0,
                'top_1_agreement': 0.0,
                'top_3_agreement': 0.0,
                'top_5_agreement': 0.0
            }

    def comprehensive_evaluation(self,
                                 text: str,
                                 attention_data: np.ndarray,
                                 attribution_data: Dict[str, np.ndarray],
                                 human_annotations: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Run comprehensive evaluation of visualization methods.
        """
        results = {}

        try:
            # Attention concentration metrics
            if attention_data.ndim >= 2:
                results['attention_concentration'] = self.attention_concentration(
                    attention_data)

            # Stability metrics
            if attention_data.ndim == 2:  # layerwise trajectories
                results['stability'] = self.stability_across_layers(
                    attention_data)

            # Method agreement
            method_pairs = []
            method_names = list(attribution_data.keys())

            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    method1, method2 = method_names[i], method_names[j]
                    agreement = self.method_agreement(
                        attribution_data[method1],
                        attribution_data[method2],
                        (method1, method2)
                    )
                    method_pairs.append({
                        'methods': f"{method1}_vs_{method2}",
                        'agreement': agreement
                    })

            results['method_agreements'] = method_pairs

            # Faithfulness evaluation for each method (simplified)
            faithfulness_scores = {}
            for method_name, scores in attribution_data.items():
                try:
                    faith_score = self.faithfulness_correlation(text, scores)
                    faithfulness_scores[method_name] = faith_score
                except Exception as e:
                    print(
                        f"Could not compute faithfulness for {method_name}: {e}")
                    faithfulness_scores[method_name] = 0.0

            results['faithfulness'] = faithfulness_scores

            # Plausibility evaluation (if human annotations available)
            if human_annotations is not None:
                plausibility_scores = {}
                for method_name, scores in attribution_data.items():
                    plaus_score = self.plausibility_human_agreement(
                        scores, human_annotations)
                    plausibility_scores[method_name] = plaus_score

                results['plausibility'] = plausibility_scores

        except Exception as e:
            print(f"Error in comprehensive evaluation: {e}")
            results['error'] = str(e)

        return results

    def _get_prediction_logits(self, text: str) -> Optional[torch.Tensor]:
        """Get model prediction logits for text."""
        if self.model is None or self.tokenizer is None:
            return None

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.logits[0]
        except Exception as e:
            print(f"Error getting prediction logits: {e}")
            return None

    def _mask_tokens(self, text: str, mask_indices: List[int]) -> Optional[str]:
        """Mask tokens at specified indices."""
        if self.tokenizer is None:
            return None

        try:
            tokens = self.tokenizer.tokenize(text)

            for idx in sorted(mask_indices, reverse=True):
                if 0 <= idx < len(tokens):
                    tokens[idx] = self.tokenizer.mask_token or "[MASK]"

            return self.tokenizer.convert_tokens_to_string(tokens)
        except Exception as e:
            print(f"Error masking tokens: {e}")
            return None


def run_benchmark_evaluation(texts: List[str],
                             extractor,
                             visualizer,
                             save_results: bool = True) -> Dict[str, any]:
    """
    Run benchmark evaluation on a set of texts with improved error handling.
    """
    print("Starting benchmark evaluation...")

    all_results = []

    for i, text in enumerate(texts):
        print(f"Evaluating text {i+1}/{len(texts)}")

        try:
            # Extract different attribution methods (simplified)
            attribution_data = {}

            # Extract attention trajectory
            try:
                tokens, attention_traj = extractor.extract_layerwise_trajectories(
                    text)
                attribution_data['attention_trajectory'] = attention_traj[-1] if attention_traj.ndim > 1 else attention_traj
            except Exception as e:
                print(f"  Error extracting attention trajectory: {e}")

            # Extract attention rollout
            try:
                _, rollout = extractor.extract_attention_rollout(text)
                attribution_data['attention_rollout'] = rollout
            except Exception as e:
                print(f"  Error extracting attention rollout: {e}")

            # Skip integrated gradients for now
            print(f"  Skipping integrated gradients due to implementation issues")

            # Basic evaluation
            if attribution_data:
                evaluator = EvaluationMetrics(
                    extractor.model_name, extractor.device)

                # Simple stability check if we have trajectory data
                stability_result = {}
                if 'attention_traj' in locals() and attention_traj.ndim == 2:
                    stability_result = evaluator.stability_across_layers(
                        attention_traj)

                result = {
                    'text': text[:100],  # Store truncated text
                    'attribution_methods': list(attribution_data.keys()),
                    'stability': stability_result
                }
            else:
                result = {
                    'text': text[:100],
                    'error': 'No attribution methods succeeded',
                    'attribution_methods': []
                }

            all_results.append(result)

        except Exception as e:
            print(f"  Error evaluating text {i+1}: {e}")
            all_results.append({
                'text': text[:100],
                'error': str(e),
                'attribution_methods': []
            })
            continue

    # Aggregate results
    successful_results = [r for r in all_results if 'error' not in r]

    aggregated = {
        'num_texts': len(texts),
        'successful_evaluations': len(successful_results),
        'individual_results': all_results,
        'summary_statistics': _aggregate_metrics(successful_results)
    }

    if save_results:
        import json
        try:
            with open('evaluation_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = _convert_numpy_to_lists(aggregated)
                json.dump(json_results, f, indent=2, default=str)
            print("✓ Evaluation results saved to evaluation_results.json")
        except Exception as e:
            print(f"✗ Error saving evaluation results: {e}")

    return aggregated


def _aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Aggregate metrics across multiple texts."""
    aggregated = {}

    # Collect stability scores
    stability_scores = []
    for result in results:
        if 'stability' in result and 'mean_layer_correlation' in result['stability']:
            score = result['stability']['mean_layer_correlation']
            if not np.isnan(score):
                stability_scores.append(score)

    if stability_scores:
        aggregated['mean_stability'] = np.mean(stability_scores)
        aggregated['std_stability'] = np.std(stability_scores)

    return aggregated


def _convert_numpy_to_lists(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_lists(item) for item in obj]
    else:
        return obj
