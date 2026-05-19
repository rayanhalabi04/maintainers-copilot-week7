# Transformer Issue Classifier Model Card

## Model

- Base model: `hf-internal-testing/tiny-random-distilbert`
- Task: GitHub issue classification
- Labels: `bug, feature, docs, question`
- Architecture: encoder transformer with a sequence classification head

## Data

- Train split: `/Users/rayanhalabi/Desktop/maintainers-copilot/data/processed/train.jsonl`
- Validation split: `/Users/rayanhalabi/Desktop/maintainers-copilot/data/processed/val.jsonl`
- Test split: `/Users/rayanhalabi/Desktop/maintainers-copilot/data/processed/test.jsonl`
- Split sizes: train=64, validation=64, test=64
- Dataset SHA-256: `8bd5457e9dc1059bb8568ffed284e32eda80f3f95b5d4d88be811b7180fa97e0`

## Hyperparameters

- Epochs: 1.0
- Learning rate: 2e-05
- Batch size: 8
- Max sequence length: 256
- Weight decay: 0.01
- Seed: 42
- Max train samples: 64
- Max eval samples: 64

## Results

### Validation

- Accuracy: 0.328125
- Macro-F1: 0.12352941176470589
- Per-class F1: `{"bug": 0.0, "docs": 0.0, "feature": 0.0, "question": 0.49411764705882355}`

### Test

- Accuracy: 0.265625
- Macro-F1: 0.10493827160493827
- Per-class F1: `{"bug": 0.0, "docs": 0.0, "feature": 0.0, "question": 0.41975308641975306}`
- Average prediction latency: 3.7522221875292416 ms/example

## Notes

This model is trained and evaluated on the same processed splits as the classical ML baseline and the LLM baseline.
