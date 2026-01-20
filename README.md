# Predictive BPM: An Attention-Based Intelligent Process Automation
Business Process Management (BPM) platforms such as PEGA and UiPath rely pre- dominantly on rule-based and static workflow definitions. While effective for structured automation, these systems are reactive in nature and lack predictive intelligence to an- ticipate delays, failures, or sub-optimal process paths.

# Attentive-LSTM for Predictive Process Mining

## 📌 Project Overview
This project implements a predictive process mining system designed to enhance Enterprise Automation tools like PEGA and UiPath. 

Traditional BPM systems primarily operate on deterministic, static rules and lack foresight, making interventions inherently reactive. This solution introduces an **Attention-augmented LSTM Architecture** to proactively predict the "Next Best Action" and "Remaining Time-to-Completion" with high interpretability.

---

## 🚀 Key Features & Modifications
### 1. Attentive-LSTM Architecture
Standard LSTM networks often act as "black boxes." This project implements an **Attentive-LSTM** to address critical industry gaps:
* **Attention Integration:** The model calculates an attention score for every event in a trace to identify which specific workflow step is the statistical driver of a bottleneck.
* **Context Awareness:** Utilizes the hidden states of the LSTM to solve "Context Insensitivity," capturing the intricate sequence of events leading to the current state.
* **Explainability:** Provides a visual heatmap of activity importance, solving the lack of transparency found in standard deep learning models.

### 2. Intelligent Process Automation (IPA)
* **Dynamic Resource Allocation:** Based on the "Next Best Action" prediction, RPA bots can be pre-allocated to high-probability paths.
* **Proactive Bottleneck Management:** Predicts delays before they occur, shifting BPM from a reactive to a predictive paradigm.

---

## 🛠️ Technical Workflow
The system follows a rigorous Data Science pipeline:
1.  **Data Ingestion:** Extraction of XES/CSV event logs using **Pm4Py**.
2.  **Pre-processing:** One-hot encoding of activity labels and normalization of timestamps using **Pandas**.
3.  **Modeling:** Architecture built with LSTM layers followed by a custom Attention layer using **PyTorch**.
4.  **Evaluation:** * **Time Prediction:** Measured via Mean Absolute Error (MAE).
    * **Activity Prediction:** Measured via Categorical Accuracy.



---

## 💻 Software Stack
* **Language:** Python 3.11
* **Process Mining:** Pm4Py (Handling event logs)
* **Deep Learning:** PyTorch (Neural Network implementation)
* **Data Processing:** NumPy & Pandas (Matrix operations and data cleaning)

---

## 📖 Literature Reference
This work builds upon the foundational research by **Tax et al. (2017)**: *"Predictive Business Process Monitoring with LSTM Neural Networks,"* which proved that LSTMs outperform traditional Markov models in capturing long-range dependencies in event logs.

