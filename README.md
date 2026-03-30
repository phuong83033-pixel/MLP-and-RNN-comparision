# ML-Midterm
This project focuses on Sentiment Analysis, comparing two fundamental architectural families: MLPs (Bag-of-Words approach) and RNNs/LSTMs/GRUs (Sequential approach).

# Sentiment Analysis on IMDb Dataset: MLP vs. RNN Architectures

## 1. Introduction
This project performs binary sentiment classification (Positive/Negative) on the IMDb Large Movie Review dataset. The core objective is to systematically compare the performance of **Multi-Layer Perceptron (MLP)**, which treats text as a bag of mean-pooled embeddings, against the **Recurrent Neural Network (RNN) family** (Vanilla RNN, LSTM, and GRU), which processes information sequentially. The study includes several ablation experiments to evaluate the impact of hyperparameters on model robustness.

## 2. Methodology
The project follows a rigorous deep learning pipeline:
* **Preprocessing:** Includes HTML tag removal, tokenization, building a 20,000-word vocabulary, and converting text into integer-ID sequences.
* **Architectures:**
    * **MLP:** Uses an embedding layer followed by mean-pooling to create a fixed-size sentence representation.
    * **RNN Family:** Utilizes `pack_padded_sequence` to handle variable-length inputs efficiently and capture long-term dependencies.
* **Ablation Studies:** Conducted using a single-variable protocol, changing exactly one hyperparameter at a time (e.g., number of layers, embedding dimension $d_e$, or dropout rate $p$) to isolate its effect on performance.

## 3. Key Findings
* **Best Model:** The **2-layer GRU** achieved the highest performance with **88.18% Accuracy** and **88.55% F1-score**.
* **Gating Importance:** LSTM and GRU outperformed Vanilla RNN by over 14% in accuracy, proving that gating mechanisms are essential to mitigate the vanishing gradient problem.
* **Negation Handling:** Recurrent models showed superior robustness on sentences containing negations (e.g., "not", "never"), with only a ~1.12% drop in accuracy compared to a ~4.35% drop for the MLP.
* **Overfitting:** MLP models exhibited higher overfitting (up to 10.27 pp gap); increasing the dropout rate to 0.5 successfully reduced this gap to 8.13 pp.

## 4. Hardware/Software Requirements
* **Hardware:** A CUDA-enabled GPU is highly recommended for training recurrent architectures.
* **Language:** Python 3.8+
* **Libraries:** * `torch`, `torchtext` (Core deep learning & NLP)
    * `scikit-learn` (Metrics & Confusion Matrix)
    * `matplotlib` (Visualization of learning curves)
    * `pandas`, `numpy` (Data manipulation)

## 5. Instructions to Run the Code

### Folder Structure
.
├── checkpoints/          
├── notebooks/                
├── src/                  
└── Readme.md 

### 6.Team Meber Contributions

NguyenMinhDuc_523C0007 :Implement MLP, Data_preprocessing, Visualization & Documentation 
NguyenHoangMinh_523C0001 : Implement RNN, Model evaluation & Documentation 
