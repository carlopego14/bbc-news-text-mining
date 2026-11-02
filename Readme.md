# BBC News Text Mining: Classification and Clustering

Can a model tell a sports article from a business one — and can it discover the news categories on its own, without labels?
This project builds a complete **text mining pipeline** on the BBC News dataset and compares **sparse (TF-IDF)** and **dense (transformer embeddings)** text representations on two tasks:

1. **Supervised classification** of articles into 5 categories
2. **Unsupervised clustering** to recover the category structure without labels

> Group project for the *Text Mining and Search* course — MSc in Data Science, University of Milano-Bicocca (2026).

---

## Key findings

- **Classification: TF-IDF wins.** TF-IDF + SVM reaches **98.0% accuracy** (macro F1 0.980), outperforming all configurations based on BERT embeddings.
- **Clustering: embeddings win.** Sentence embeddings + K-Means find exactly **k = 5** clusters (by silhouette analysis), closely matching the 5 real BBC categories — without ever seeing the labels.
- **Typical errors** happen between *Business* and *Politics*, which share economic vocabulary; *Sport* is the easiest category (100% recall).
- **No representation is best everywhere**: distinctive keywords favour TF-IDF for classification, while embeddings capture the high-level semantic structure needed for clustering.

| Representation | Model | Accuracy | Macro F1 |
|---|---|---|---|
| TF-IDF | SVM | **0.980** | **0.980** |
| TF-IDF | Random Forest | 0.980 | 0.979 |
| TF-IDF | Logistic Regression | 0.978 | 0.977 |
| BERT | SVM | 0.973 | 0.973 |
| BERT | Random Forest | 0.971 | 0.970 |
| BERT | Logistic Regression | 0.944 | 0.944 |

<!-- Add a figure here, e.g. the confusion matrix or the PCA cluster plot:
<p align="center"><img src="images/pca_clusters.png" width="80%" alt="PCA of document embeddings: predicted clusters vs true labels"></p>
-->

---

## Data

**BBC News dataset**: news articles labelled in 5 topical categories — *Business, Entertainment, Politics, Sport, Tech*.
The raw dataset is publicly available ([BBC full text dataset](http://mlg.ucd.ie/datasets/bbc.html)) and is not included in this repository.

---

## Methods

### Preprocessing
Two pipelines, each tailored to its representation:
- **For TF-IDF**: lowercasing, removal of punctuation, special characters and numbers, tokenization, stopword removal, lemmatization.
- **For embeddings**: light cleaning only, to preserve the context that transformer models rely on.

### Text representation
- **TF-IDF** sparse vectors
- **Sentence embeddings** from a pretrained transformer model (sentence-transformers)

### Classification
- Logistic Regression, SVM (RBF kernel), Random Forest
- Stratified 80/20 train-test split; evaluation with accuracy, macro F1 and confusion matrices

### Clustering
- K-Means and Agglomerative Clustering on both representations
- Number of clusters chosen by silhouette analysis (k from 2 to 20, cosine distance)
- Evaluation with internal (silhouette) and external metrics (ARI, NMI, V-measure, purity, mapped macro F1)
- PCA visualization and top TF-IDF terms per cluster for interpretability

---

## Repository structure

```
├── notebooks/
│   ├── 01_data_loading_eda.ipynb       # load dataset, exploratory analysis
│   ├── 02_preprocessing.ipynb          # two preprocessing pipelines
│   ├── 03_text_representation.ipynb    # TF-IDF and sentence embeddings
│   ├── 04_text_classification.ipynb    # LR, SVM, RF comparison
│   └── 05_text_clustering.ipynb        # K-Means, agglomerative, evaluation
├── images/
├── report/
│   ├── report.pdf
│   └── slides.pdf
├── requirements.txt
└── README.md
```

**How to run:** download the dataset into `data/raw/`, then run the notebooks in order — each one saves the outputs used by the next (in `data/processed/`). The notebooks run both locally and on Google Colab.

---

## Limitations

- The BBC dataset is small and clean, with well-separated categories: results would likely be lower on noisier, real-world news.
- Embeddings were used as frozen features, without fine-tuning on the task.

---

## Tools

Python · pandas · scikit-learn · sentence-transformers · matplotlib · seaborn · Google Colab

---

## Authors

- **Carlo Pegoraro** — [GitHub](https://github.com/carlopego14)
- **Davide Fabio Loreti**
- **Alberto Cera**

### Contributions

| Part | Author |
|---|---|
| Data loading, EDA and preprocessing | Alberto Cera |
| Text representation and classification | Davide Fabio Loreti |
| Clustering, evaluation and visualizations | Carlo Pegoraro |
