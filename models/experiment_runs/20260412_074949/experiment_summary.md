# Experiment Suite Summary

## Run Outcomes

- **full_mlp**: success | test_acc=0.9130434989929199 | notes=Strong baseline performance
- **full_lstm**: success | test_acc=0.1304347813129425 | notes=Near chance-level test accuracy
- **full_lstm_motion**: success | test_acc=0.47826087474823 | notes=Run completed
- **tiny_overfit_lstm**: success | test_acc=None | notes=Tiny overfit train accuracy=1.0000
- **tiny_overfit_mlp**: success | test_acc=None | notes=Tiny overfit train accuracy=1.0000

## Brief Comparison

- Current best baseline in this suite: **full_mlp** (mlp) with test accuracy 0.9130434989929199.
- Failed or near-chance runs:
  - full_lstm: success (Near chance-level test accuracy)
- Interpretation: the MLP remains the strongest valid baseline unless a sequence model exceeds it.
