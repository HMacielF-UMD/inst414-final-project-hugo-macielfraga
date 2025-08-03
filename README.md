# INST414 Final Project – Hugo Maciel Fraga

## Project Overview

This project analyzes and predicts the mood of songs based on their audio features.  
The goal is to explore both **supervised** and **unsupervised** machine learning models to classify songs as either **Happy** or **Sad**.

### Business Problem
In today's music platforms, understanding a listener's mood is essential for engagement. This project aims to automate mood classification using audio features, which could support features like mood-based playlists or emotional tracking.

### Datasets Used
- `mood_relation.csv`: manually labeled song moods (Happy or Sad).
- Audio features extracted using `librosa` from downloaded YouTube/Spotify tracks.

### Techniques Employed
- **Feature Extraction:** Librosa (tempo, MFCC, chroma, ZCR, etc.)
- **Supervised Model:** Random Forest Classifier
- **Unsupervised Model:** KMeans Clustering
- **Evaluation:** Classification metrics and cluster distribution
- **Visualization:** Seaborn and Matplotlib heatmaps, confusion matrices, and bar charts

### Expected Outputs
- CSV files with predicted moods and clusters
- Evaluation reports
- Visualizations exported to `data/vis/`

---

## Setup Instructions

### Clone the repository

```bash
git clone https://github.com/[your-username]/inst414-final-project-hugo-macielfraga.git
cd inst414-final-project-hugo-macielfraga
```

### Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate.bat    # On Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run each part of the project as needed:

1. **Feature Extraction**

```bash
python etl/extract.py
```

2. **Transform Features**

```bash
python etl/transform.py
```

3. **Supervised Modeling**

```bash
python analysis/supervised_model.py
```

4. **Unsupervised Modeling**

```bash
python analysis/unsupervised_model.py
```

5. **Evaluation**

```bash
python analysis/evaluate.py
```

6. **Visualizations**

```bash
python vis/visualizations.py
```

---

## Code Package Structure

```text
inst414-final-project-hugo-macielfraga/
│
├── data/
│   ├── extracted/              # Raw audio features
├── processed/              # Cleaned and merged datasets
│   ├── outputs/                # Predictions and clustering results
│   ├── provided/               # mood_relation.csv
│   ├── reference-tables/      # Data dictionary and any lookup tables
│   └── vis/                    # Exported visualizations
│
├── etl/
│   ├── extract.py              # Download and extract audio features
│   ├── transform.py            # Clean and structure data
│
├── analysis/
│   ├── supervised_model.py     # Train and evaluate a classifier
│   ├── unsupervised_model.py   # Cluster analysis
│   └── evaluate.py             # Summary of results
│
├── vis/
│   └── visualizations.py       # Generate charts and graphs
│
├── main.py                     
├── requirements.txt
└── README.md
```
