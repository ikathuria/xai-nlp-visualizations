# XAI NLP Visualization Setup Guide

## Quick Start

1. **Clone or create project directory**
```bash
mkdir xai-nlp-visualization
cd xai-nlp-visualization
```

2. **Create virtual environment**
```bash
python -m venv xai_env
source xai_env/bin/activate  # Linux/Mac
# OR
xai_env\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the demo**
```bash
python main_demo.py
```

## Project Structure
```
xai-nlp-visualization/
├── src/
│   ├── trajectory.py      # Enhanced trajectory extraction
│   ├── visualization.py   # Advanced plotting functions
│   ├── evaluation_metrics.py      # Evaluation and benchmarking
│   └── utils.py                   # Utility functions
├── figures/                       # Generated visualizations
├── reports/                       # Analysis reports
├── data/                         # Saved results and data
├── main_demo.py                  # Main demonstration script
├── requirements.txt              # Python dependencies
└── README.md                     # This file

## Usage Examples

### Basic Usage
```python
from trajectory import EnhancedTrajectoryExtractor
from visualization import AdvancedVisualizer

# Initialize
extractor = EnhancedTrajectoryExtractor("bert-base-uncased")
visualizer = AdvancedVisualizer()

# Analyze text
text = "I love this research paper!"
tokens, trajectories = extractor.extract_layerwise_trajectories(text)

# Visualize with different modes
fig = visualizer.plot_layerwise_trajectories(
    tokens, trajectories, 
    mode="exclude",  # Remove [CLS]/[SEP]
    save_path="my_analysis.png"
)
```

### Advanced Analysis
```python
from main_demo import XAIVisualizationPipeline

# Full pipeline
pipeline = XAIVisualizationPipeline("distilbert-base-uncased-finetuned-sst-2-english")

# Comprehensive analysis
results = pipeline.analyze_single_text(
    "This movie is amazing!", 
    methods=['cls_attention', 'attention_rollout', 'integrated_gradients'],
    modes=['full', 'exclude', 'downweight']
)

# Generate publication figures
figure_paths = pipeline.create_publication_figures([
    "Positive sentiment example",
    "Negative sentiment example"
])
```

### Evaluation and Benchmarking
```python
from evaluation_metrics import EvaluationMetrics

# Initialize evaluator
evaluator = EvaluationMetrics("bert-base-uncased")

# Evaluate faithfulness
faithfulness_score = evaluator.faithfulness_correlation(
    text="Your text here",
    attributions=attribution_scores
)

# Run comprehensive evaluation
results = evaluator.comprehensive_evaluation(
    text, attention_data, attribution_data
)
```

## Configuration Options

### Model Support
- BERT variants: `bert-base-uncased`, `bert-large-uncased`
- RoBERTa variants: `roberta-base`, `roberta-large`  
- DistilBERT: `distilbert-base-uncased`
- Task-specific models: `distilbert-base-uncased-finetuned-sst-2-english`

### Visualization Modes
- **Full**: Show all tokens including [CLS] and [SEP]
- **Exclude**: Remove special tokens completely
- **Downweight**: Reduce special token importance by factor

### Attribution Methods
- **CLS Attention**: Attention from [CLS] token to all others
- **Attention Rollout**: Recursive attention flow analysis
- **Integrated Gradients**: Gradient-based token attribution
- **Multi-head Analysis**: Individual attention head patterns

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size or use CPU: `device="cpu"`
   - Use smaller model: `distilbert-base-uncased`

2. **Import errors**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

3. **Visualization not showing**
   - For Jupyter: add `%matplotlib inline`
   - For interactive plots: ensure plotly installed

4. **Model download issues**
   - Check internet connection
   - Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`

### Performance Tips

1. **Use GPU acceleration**
```python
pipeline = XAIVisualizationPipeline("bert-base-uncased", device="cuda")
```

2. **Batch processing**
```python
results = pipeline.run_benchmark_study(texts, save_results=True)
```

3. **Memory optimization**
```python
# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run code formatting: `black .`
5. Submit pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{xai_nlp_viz_2025,
  title={XAI for NLP via Parameterized Visualization},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```
