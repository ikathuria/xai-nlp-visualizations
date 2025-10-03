# XAI Visualization Pipeline - Demo Report
============================================================

**Model**: distilbert-base-uncased-finetuned-sst-2-english
**Number of examples**: 4
**Analysis date**: 2025-09-30 17:09:43

## Example 1
**Text**: I love this movie! The acting was fantastic and the plot was engaging....
**Methods used**: cls_attention, attention_rollout, integrated_gradients
**Generated figures**:
  - full_trajectory: figures/trajectories_0_full.png
  - exclude_trajectory: figures/trajectories_0_exclude.png
  - downweight_trajectory: figures/trajectories_0_downweight.png
  - method_comparison: figures/method_comparison_0.png

## Example 2
**Text**: This product is terrible. Poor quality and overpriced....
**Methods used**: cls_attention, attention_rollout, integrated_gradients
**Generated figures**:
  - full_trajectory: figures/trajectories_1_full.png
  - exclude_trajectory: figures/trajectories_1_exclude.png
  - downweight_trajectory: figures/trajectories_1_downweight.png
  - method_comparison: figures/method_comparison_1.png

## Example 3
**Text**: The research paper presents interesting findings about neural networks....
**Methods used**: cls_attention, attention_rollout, integrated_gradients
**Generated figures**:
  - full_trajectory: figures/trajectories_2_full.png
  - exclude_trajectory: figures/trajectories_2_exclude.png
  - downweight_trajectory: figures/trajectories_2_downweight.png
  - method_comparison: figures/method_comparison_2.png

## Example 4
**Text**: Climate change poses significant challenges for future generations....
**Methods used**: cls_attention, attention_rollout, integrated_gradients
**Generated figures**:
  - full_trajectory: figures/trajectories_3_full.png
  - exclude_trajectory: figures/trajectories_3_exclude.png
  - downweight_trajectory: figures/trajectories_3_downweight.png
  - method_comparison: figures/method_comparison_3.png