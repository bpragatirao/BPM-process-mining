## 📊 Processed Dataset Description

### **Data Source**
* **Source Name:** BPI Challenge 2017 Event Log
* **Domain:** Loan Application Process (Business Process Management)

### **Preprocessing Pipeline**
To prepare the raw data for the LSTM network, the following steps were performed:
* **Format Conversion:** Event logs were converted from **XES** (Extensible Event Stream) to a structured tabular format.
* **Feature Extraction:** Isolated critical process attributes: `case_id`, `activity`, and `timestamp`.
* **Chronological Sorting:** Events were sorted by `timestamp` within each individual case to maintain the temporal sequence of the process.

### **Prefix Generation & Sequence Modeling**
For **Next-Activity Prediction**, the event sequences were transformed into prefix traces:
* **Max Prefix Length:** 15 (fixed window size for the LSTM memory).
* **Dataset Scale:** * **Cases utilized:** 31,509 unique process instances.
    * **Total Samples generated:** 112,440 prefixes.

### **Data Encoding**
* **Activity Encoding:** Applied **Integer Encoding** to categorical activity labels to convert them into numerical vectors.
* **Padding:** Pre-padding was applied to ensure all input sequences met the uniform length of 15, allowing for batch processing in the neural network.


### **Output Artifacts**
The preprocessing script generates the following files for model training:
| File Name | Description |
| :--- | :--- |
| `sequences.csv` | Encoded activity prefixes used as model features (X). |
| `labels.csv` | The "ground truth" next activity labels (y). |