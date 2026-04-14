# Experiment Suite Summary

## Run Outcomes

- **full_mlp**: success | test_acc=0.8840579986572266 | notes=Strong baseline performance
- **full_mlp_motion**: success | test_acc=0.8115941882133484 | notes=Run completed
- **full_lstm_motion_valacc**: success | test_acc=0.5362318754196167 | notes=Run completed
- **full_gru_motion**: success | test_acc=0.07246376574039459 | notes=Near chance-level test accuracy

## Brief Comparison

- Current best baseline in this suite: **full_mlp** (mlp) with test accuracy 0.8840579986572266.
- Failed or near-chance runs:
  - full_gru_motion: success (Near chance-level test accuracy)
- Interpretation: the MLP remains the strongest valid baseline unless a sequence model exceeds it.
