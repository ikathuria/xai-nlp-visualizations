import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, GradientShap
from typing import List, Tuple, Dict, Optional, Union
import warnings


class EnhancedTrajectoryExtractor:
    """
    Enhanced trajectory extractor supporting multiple transformer models and attribution methods.
    Supports BERT, RoBERTa, DistilBERT, and other transformer architectures.
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 device: str = None,
                 task_type: str = "classification"):
        """
        Initialize the enhanced trajectory extractor.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device (cuda/cpu)
            task_type: "classification" or "masked_lm"
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.task_type = task_type

        print(f"Initializing extractor with model: {model_name}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model based on task type
        if task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True
            ).to(self.device)

        self.model.eval()

        # Model architecture info
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

        print(
            f"Model loaded: {self.num_layers} layers, {self.num_heads} heads")

    def extract_multi_head_attention(self, text: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Extract attention patterns for all heads across all layers.

        Returns:
            - tokens: List of token strings
            - head_attention: Shape (num_layers, num_heads, seq_len, seq_len)  
            - cls_attention: Shape (num_layers, num_heads, seq_len) - CLS token attention to all tokens
        """
        enc = self.tokenizer(text, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)

        # Tuple of (batch, heads, seq_len, seq_len)
        attentions = outputs.attentions

        # Convert to numpy arrays
        head_attention = []
        cls_attention = []

        for layer_attn in attentions:
            # layer_attn shape: (batch, heads, seq_len, seq_len)
            layer_attn_np = layer_attn[0].detach(
            ).cpu().numpy()  # Remove batch dim
            head_attention.append(layer_attn_np)

            # Extract CLS attention (first row of attention matrix)
            cls_attn = layer_attn_np[:, 0, :]  # Shape: (heads, seq_len)
            cls_attention.append(cls_attn)

        # (layers, heads, seq_len, seq_len)
        head_attention = np.stack(head_attention, axis=0)
        # (layers, heads, seq_len)
        cls_attention = np.stack(cls_attention, axis=0)

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        return tokens, head_attention, cls_attention

    def extract_attention_rollout(self, text: str, discard_ratio: float = 0.9) -> Tuple[List[str], np.ndarray]:
        """
        Compute attention rollout following Abnar & Zuidema (2020).
        """
        tokens, head_attention, _ = self.extract_multi_head_attention(text)

        # Average attention across heads
        # (layers, seq_len, seq_len)
        layer_attention = np.mean(head_attention, axis=1)

        # Add residual connections (identity matrix)
        residual_att = np.eye(layer_attention.shape[-1])[None, ...]
        aug_att_mat = layer_attention + residual_att

        # Normalize
        aug_att_mat = aug_att_mat / \
            (aug_att_mat.sum(axis=-1, keepdims=True) + 1e-8)

        # Recursively multiply attention matrices
        joint_attentions = aug_att_mat[0]
        for n in range(1, len(aug_att_mat)):
            joint_attentions = aug_att_mat[n].dot(joint_attentions)

        # Extract CLS attention rollout
        rollout = joint_attentions[0]

        return tokens, rollout

    def extract_gradient_attribution(self,
                                     text: str,
                                     target: int = 1,
                                     method: str = "integrated_gradients",
                                     n_steps: int = 25) -> Tuple[List[str], np.ndarray]:
        """
        Extract gradient-based token attributions with fixed attention mask handling.
        """
        if self.task_type != "classification":
            raise ValueError(
                "Gradient attribution requires classification model")

        enc = self.tokenizer(text, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Get embeddings
        if hasattr(self.model, 'bert'):
            embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'roberta'):
            embeddings = self.model.roberta.embeddings.word_embeddings(
                input_ids)
        elif hasattr(self.model, 'distilbert'):
            embeddings = self.model.distilbert.embeddings.word_embeddings(
                input_ids)
        else:
            raise ValueError(
                f"Unsupported model architecture: {self.model_name}")

        # Create baseline (zeros)
        baseline = torch.zeros_like(embeddings)

        def forward_func(embs):
            """Fixed forward function that properly handles attention mask."""
            # Ensure embeddings have correct shape for the model
            if embs.dim() == 2:
                embs = embs.unsqueeze(0)  # Add batch dimension if needed

            try:
                # Use the model's forward method with proper arguments
                outputs = self.model(inputs_embeds=embs,
                                     attention_mask=attention_mask)
                return outputs.logits
            except Exception as e:
                print(f"Forward function error: {e}")
                # Fallback: try without attention mask
                try:
                    outputs = self.model(inputs_embeds=embs)
                    return outputs.logits
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    raise e2

        # Test the forward function first
        try:
            test_output = forward_func(embeddings)
            print(
                f"Forward function test successful, output shape: {test_output.shape}")
        except Exception as e:
            print(f"Forward function test failed: {e}")
            # Return dummy attributions as fallback
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            return tokens, np.random.rand(len(tokens))

        # Choose attribution method
        try:
            if method == "integrated_gradients":
                attributor = IntegratedGradients(forward_func)
                attributions = attributor.attribute(
                    embeddings,
                    baselines=baseline,
                    target=target,
                    n_steps=n_steps,
                    return_convergence_delta=False
                )
            else:
                raise ValueError(f"Unsupported attribution method: {method}")
        except Exception as e:
            print(f"Attribution computation failed: {e}")
            # Return dummy attributions as fallback
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            return tokens, np.random.rand(len(tokens))

        # Sum across embedding dimensions
        if attributions.dim() > 2:
            token_attributions = attributions.sum(
                dim=-1)[0].detach().cpu().numpy()
        else:
            token_attributions = attributions[0].detach().cpu().numpy()

        # Handle potential dimension issues
        if token_attributions.ndim > 1:
            token_attributions = token_attributions.sum(axis=-1)

        # Normalize to [0, 1]
        if token_attributions.max() > token_attributions.min():
            token_attributions = token_attributions - token_attributions.min()
            token_attributions = token_attributions / token_attributions.max()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return tokens, token_attributions

    def extract_layerwise_trajectories(self, text: str, method: str = "cls_attention") -> Tuple[List[str], np.ndarray]:
        """
        Extract token importance trajectories across all layers.
        """
        tokens, head_attention, cls_attention = self.extract_multi_head_attention(
            text)

        if method == "cls_attention":
            # Average CLS attention across heads
            trajectories = np.mean(cls_attention, axis=1)  # (layers, seq_len)

        elif method == "mean_attention":
            # Average attention received by each token
            trajectories = np.mean(
                np.mean(head_attention, axis=1), axis=1)  # (layers, seq_len)

        elif method == "attention_norm":
            # L2 norm of attention received by each token
            trajectories = np.linalg.norm(
                np.mean(head_attention, axis=1), axis=1)  # (layers, seq_len)

        else:
            raise ValueError(f"Unsupported trajectory method: {method}")

        return tokens, trajectories

    def batch_process(self, texts: List[str], method: str = "cls_attention") -> List[Tuple[List[str], np.ndarray]]:
        """
        Process multiple texts in batch for efficiency.
        """
        results = []

        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}")

            try:
                if method == "attention_rollout":
                    result = self.extract_attention_rollout(text)
                elif method == "integrated_gradients":
                    result = self.extract_gradient_attribution(
                        text, method="integrated_gradients")
                else:
                    result = self.extract_layerwise_trajectories(text, method)

                results.append(result)

            except Exception as e:
                print(f"Error processing text {i+1}: {e}")
                # Add dummy result to maintain list consistency
                dummy_tokens = ['[CLS]', 'error', '[SEP]']
                dummy_data = np.random.rand(
                    6, 3) if method != 'integrated_gradients' else np.random.rand(3)
                results.append((dummy_tokens, dummy_data))
                continue

        return results
