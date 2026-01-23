# Model Comparison Summary

This directory contains trained models and evaluation artifacts for the **Predictive Business Process Monitoring** project. Two sequence prediction models were trained and compared for the task of **next-activity prediction** using business process event logs.

## Models Evaluated

1. **Baseline LSTM**
   - Standard LSTM-based sequence model
   - Treats all past events in a process trace uniformly

2. **LSTM with Attention Mechanism**
   - Extends the baseline LSTM with an attention layer
   - Learns to focus on more relevant past activities in a trace

## Key Observations

The Attention-based LSTM achieved **higher prediction accuracy** and **lower loss values** compared to the baseline LSTM. This improvement is attributed to the model’s ability to selectively emphasize important historical events in the execution trace, rather than assigning equal importance to all past activities.

## Conclusion

The experimental results validate the hypothesis that **context-aware dependency modeling**, enabled through attention mechanisms, improves predictive performance in business process monitoring tasks. This demonstrates the effectiveness of attention-based architectures for modeling complex, long-running business processes.
