# QUICK LIVE CODE DEMONSTRATION & EMERGENCY REFERENCE GUIDE
**Project:** Explainable Hybrid Deep Learning-Based Network Intrusion Detection System  
**Teammates:** Amrita Raj (Student 1 - DL Engine) & Midhila M (Student 2 - Live Demo)  
**Academic Context:** 5th Semester Deep Learning Elective Mini-Project | Amrita Vishwa Vidyapeetham  
**Target Threat Focus:** Denial of Service (DoS) Intrusion Detection  

---

## ⚡ 1. The 10-Second Live Execution Command

Open VS Code in `Explainable_Hybrid_IDS_Final` (or `Explainable_Hybrid_IDS`), open the integrated terminal (`Ctrl + ~`), and run:

```powershell
python demo_inference.py --category DoS
```

### What Happens When You Run This:
1. **Model Loading (0.20s):** Loads trained 1D-CNN + BiLSTM + Attention (`hybrid_ids_model.keras`) and preprocessor (`preprocessing_pipeline.joblib`).
2. **Data Ingestion:** Extracts 5 real consecutive network packet flows from `dataset/benchmark_sample.csv` matching DoS.
3. **Sequential Transformation:** Constructs a 3D sequential tensor: `(Batch=1, TimeSteps=5, Features=42)`.
4. **Forward Pass (37 ms):** Runs neural forward pass across CNN, BiLSTM, and Temporal Attention.
5. **Softmax Output:** Prints the complete 10-class probability distribution with ASCII bars.
6. **Attention Weights:** Prints learned weights $\alpha_1 \dots \alpha_5$ showing which observation in the 5-step window drove the alert ($t_5$ has >90% focus!).
7. **SHAP Attribution:** Computes PermutationExplainer Shapley values (`max_evals=500`) over 210 flattened features.
8. **Export Plot:** Automatically saves publication-quality bar chart to:
   `results/shap_plots/demo_shap_attribution.png`

---

## 📂 2. Exact Files to Open in VS Code (In Order)

| Order | File Path | Important Function / Class | Line # | What to Point At & Highlight |
|---|---|---|---|---|
| **1** | `demo_inference.py` | `run_live_demonstration()` | L75–L135 | Show the clean script that runs the entire pipeline in terminal without Streamlit. |
| **2** | `preprocessing/pipeline.py` | `fit_transform_splits()` | L48–L72 | Point to strictly fitting encoders/scalers on training data (No Data Leakage). |
| **3** | `preprocessing/sequence_builder.py` | `create_sequences()` | L21–L45 | Point to the sliding window loop creating 3D tensor: `(B, 5, 42)`. |
| **4** | `models/cnn_model.py` | `build_cnn_model()` | L19–L35 | Point to `Conv1D(64, kernel=3, padding="same")` extracting local spatial packet correlations. |
| **5** | `models/bilstm_model.py` | `build_bilstm_model()` | L19–L35 | Point to `Bidirectional(LSTM(64, return_sequences=True))` producing 128 hidden units. |
| **6** | `models/attention.py` | `TemporalAttention.call()` | L24–L65 | Point to Bahdanau alignment score $e_t = v^T \tanh(W h_t + b)$ and Softmax weights $\alpha_t$. |
| **7** | `models/hybrid_model.py` | `build_hybrid_ids_model()` | L23–L65 | Point to CNN $\to$ BiLSTM $\to$ Attention $\to$ Dense(64) $\to$ Softmax(10) combined architecture. |
| **8** | `detection/predict_attack.py` | `predict_dataframe_sequence()` | L118–L185 | Point to live forward pass, latency recording (37 ms), and attention weight extraction. |
| **9** | `explainability/shap_explainer.py` | `explain_sequence_prediction()`| L110–L165 | Point to `sample_flat` (210 features), `max_evals=500`, and temporal sum aggregation $\sum_{t=1}^5 \Phi_{t,d}$. |
| **10**| `results/shap_plots/demo_shap_attribution.png` | Matplotlib Export | Image | Show crimson bars (positive threat drivers: `proto`, `ct_srv_src`) and green bars (mitigators). |
| **11**| `results/plots/confusion_matrix.png` | Confusion Matrix | Image | Point to main diagonal: 2,340 Normal correct (97.4% precision), 2,234 Generic correct (98.2% precision). |
| **12**| `results/plots/model_comparison.png` | Ablation Study | Image | Point to Weighted F1 jump: CNN (41.78%) $\to$ BiLSTM (52.42%) $\to$ Hybrid (67.65%). |

---

## 🎙️ 3. The 7-Minute Speaking Script

### Minute 0 to 2 — Architecture & Preprocessing (Student 1: Amrita)
> *"Good morning Ma'am. For our Deep Learning mini-project, we developed an Explainable Hybrid Deep Learning Network Intrusion Detection System. In `pipeline.py`, we prevent data leakage by fitting our encoders and scalers strictly on training data. In `sequence_builder.py`, we convert tabular flows into sequential sliding windows of shape `(Batch, 5, 42)` because attacks like DoS develop across consecutive packet bursts rather than single isolated packets."*

### Minute 2 to 4 — Deep Learning Layers & Attention (Student 1: Amrita)
> *"In `hybrid_model.py`, our neural head combines three specialized mechanisms: A 1D-CNN with 64 filters scans packet metrics to extract spatial correlations between packet sizes, TTL, and TCP handshakes. Next, a Bidirectional LSTM with 128 hidden units captures both chronological and reverse temporal flow transitions. Finally, in `attention.py`, our custom Bahdanau Temporal Attention layer learns dynamic weights that sum to 1.0, focusing the network on the exact packet where the intrusion peaked."*

### Minute 4 to 6 — Live Terminal Demo & Output Interpretation (Student 2: Midhila)
> *(Student 2 enters `python demo_inference.py --category DoS` in terminal)*  
> *"Now Ma'am, I am executing our live inference script in the terminal. In just 0.20 seconds, it loads the model and preprocessor, extracts 5 real flow records from UNSW-NB15, and completes the neural forward pass in 37 milliseconds! Looking at the terminal, the model classified the traffic as DoS with full 10-class probabilities. Time-step $t_5$ received over 90% attention weight. Opening `demo_shap_attribution.png`, our game-theoretic SHAP explainer shows that anomalous protocol flags and high connection counts were the crimson drivers pushing the decision toward DoS."*

### Minute 6 to 7 — Metrics & Conclusion (Student 2: Midhila)
> *"Looking at our confusion matrix (`results/plots/confusion_matrix.png`), our system achieves 97.4% precision on Normal traffic and 98.2% on Generic attacks across 9,996 test sequences. Our ablation study proves that adding BiLSTM and Attention increased Weighted F1 from 41.7% to 67.65%, demonstrating the empirical necessity of sequential modeling. We are now ready for your questions."*

---

## 🎯 4. Top 10 Most Likely Viva Questions & Instant Answers

1. **Q: Why use a 1D-CNN instead of treating all features as an LSTM input directly?**  
   *A:* Conv1D acts as a spatial filter across the 42 packet attributes, extracting localized interactions (like `sbytes` and `dur`) before feeding high-level representations to BiLSTM.

2. **Q: Why Bidirectional LSTM?**  
   *A:* It reads the 5-packet sequence forward ($t_1 \to t_5$) and backward ($t_5 \to t_1$), providing 128 hidden state dimensions that capture full temporal context.

3. **Q: What is the exact difference between Attention and SHAP?**  
   *A:* Attention operates on the **time axis** (which packet observation $t_1 \dots t_5$ in the window mattered). SHAP operates on the **feature axis** (which network metrics like `proto` or `rate` drove the classification).

4. **Q: What is the sequence shape?**  
   *A:* `(Batch, 5, 42)` — 5 consecutive flow records, each described by 42 normalized features.

5. **Q: Why is accuracy alone not enough?**  
   *A:* UNSW-NB15 is heavily imbalanced (Normal has thousands of samples while Worms has only 7). A naive model predicting Normal achieves high accuracy but 0% security. We report Weighted F1 (67.65%) and Macro F1 (35.82%).

6. **Q: How does the system detect DoS?**  
   *A:* DoS traffic displays microsecond duration, high packet rates, uniform small payload sizes, and spikes in `ct_srv_src` (connections to service).

7. **Q: Why did you set `max_evals=500` in SHAP?**  
   *A:* PermutationExplainer flattens the input to $5 \times 42 = 210$ features. SHAP requires $\text{max\_evals} \ge 2 \times 210 + 1 = 421$. Setting 500 guarantees mathematically sound evaluations without errors.

8. **Q: How fast is your model during inference?**  
   *A:* Forward pass takes **37.8 milliseconds** on CPU (0.44 ms per individual flow sample).

9. **Q: Is this system sniffing live hardware network packets?**  
   *A:* No. It ingests pre-recorded bidirectional flow records from UNSW-NB15. True network flow features require connection termination and bidirectional summary calculation by flow exporters like Argus or Zeek.

10. **Q: How does class weighting help?**  
    *A:* In `training/class_weights.py`, minority classes receive higher loss multipliers, preventing the model from ignoring rare attacks during backpropagation.
