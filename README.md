# Anuran Calls Classification

A machine learning project that classifies frog species using Mel-Frequency Cepstral Coefficients (MFCCs) from audio recordings. This project demonstrates multi-class and multi-label classification techniques using Support Vector Machines and advanced clustering methods.

## 🐸 Project Overview

This project tackles the fascinating challenge of automated species identification through bioacoustic analysis. Using MFCCs extracted from frog call recordings, we build classifiers to identify frog species at multiple taxonomic levels (Family, Genus, Species), enabling automated biodiversity monitoring and conservation efforts.

## 📊 Dataset

- **Source**: Anuran Calls (MFCCs) Dataset
- **Samples**: 7,195 frog call recordings
- **Features**: 22 Mel-Frequency Cepstral Coefficients (MFCCs)
- **Target Variables**: 
  - **Family**: 4 families (Leptodactylidae, Dendrobatidae, Hylidae, Bufonidae)
  - **Genus**: 8 genera (Adenomera, Ameerega, Dendropsophus, etc.)
  - **Species**: 10 species (AdenomeraAndre, Ameeregatrivittata, etc.)
- **Data Quality**: No missing values, no duplicates

### Taxonomic Distribution
- **Leptodactylidae**: Most diverse family with multiple genera
- **Hylidae**: Tree frogs with distinct acoustic signatures
- **Bufonidae**: True toads with characteristic calls
- **Dendrobatidae**: Poison dart frogs with unique vocalizations

## 🛠️ Technical Approach

### Data Preprocessing
- **Feature scaling** with StandardScaler for SVM optimization
- **Label encoding** for multi-class classification
- **Train-test split** (70-30) with stratified sampling
- **Data quality assessment** including missing value analysis

### Machine Learning Models

1. **Support Vector Machines (SVM)**
   - **Multi-class classification** using One-vs-Rest strategy
   - **Multi-label classification** for simultaneous prediction
   - **Hyperparameter tuning** with GridSearchCV
   - **Kernel selection**: RBF, Linear, and Polynomial kernels

2. **Evaluation Metrics**
   - **Exact Match**: Strict metric for multi-label accuracy
   - **Hamming Score/Loss**: Fraction-based multi-label evaluation
   - **Precision, Recall, F1-Score**: Per-class performance metrics
   - **Silhouette Score**: Clustering quality assessment

3. **Advanced Techniques**
   - **SMOTE**: Synthetic Minority Oversampling for class imbalance
   - **K-Means Clustering**: Unsupervised species grouping
   - **Hierarchical Clustering**: Dendrogram visualization
   - **Cross-validation**: Robust performance estimation

### Multi-Label Classification Strategy
- **Binary Relevance**: Train separate classifier for each label
- **Label Combination**: Handle multiple simultaneous predictions
- **Threshold Optimization**: Balance precision and recall

## 📈 Key Insights

### Acoustic Feature Analysis
- **MFCCs 1-5**: Capture fundamental frequency characteristics
- **MFCCs 6-15**: Represent formant structure and harmonics
- **MFCCs 16-22**: Encode fine-grained spectral details

### Classification Performance
- **Family Level**: Highest accuracy due to clear acoustic differences
- **Genus Level**: Moderate performance with some confusion
- **Species Level**: Most challenging due to subtle acoustic variations

### Clustering Results
- **K-Means**: Reveals natural groupings in acoustic space
- **Hierarchical**: Shows taxonomic relationships through dendrograms
- **Silhouette Analysis**: Validates cluster quality and optimal K

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anuran-calls-classification.git
cd anuran-calls-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
jupyter notebook notebooks/anuran_calls_analysis.ipynb
```

## 📁 Project Structure

```
anuran-calls-classification/
├── data/
│   └── Anuran Calls/
│       ├── Frogs_MFCCs.csv      # Main dataset with MFCCs
│       └── Frogs_MFCCs.txt      # Dataset description
├── notebooks/
│   └── Payapulli_Joshua_HW7.ipynb  # Complete analysis notebook
├── src/                         # Source code (if modularized)
├── results/                     # Generated plots and results
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🔍 Methodology Highlights

1. **Multi-Label Learning**: Simultaneous prediction of multiple taxonomic levels
2. **SVM Optimization**: Grid search for optimal hyperparameters
3. **Class Imbalance**: SMOTE for handling uneven species distribution
4. **Unsupervised Learning**: Clustering for species discovery
5. **Bioacoustic Analysis**: MFCC-based acoustic feature extraction

## 📊 Results Summary

### Classification Performance
- **Family Classification**: High accuracy (>90%) due to distinct acoustic signatures
- **Genus Classification**: Moderate performance with some inter-genus confusion
- **Species Classification**: Challenging task requiring fine-grained acoustic analysis

### Multi-Label Evaluation
- **Exact Match**: Strict accuracy for complete label set prediction
- **Hamming Score**: Flexible metric for partial label correctness
- **Per-Label Metrics**: Individual performance for each taxonomic level

### Clustering Analysis
- **K-Means**: Identifies natural acoustic groupings
- **Hierarchical**: Reveals taxonomic relationships
- **Silhouette Score**: Validates cluster quality

## 🎯 Applications

- **Biodiversity Monitoring**: Automated species identification in field recordings
- **Conservation Biology**: Population monitoring and habitat assessment
- **Ecological Research**: Behavioral studies and acoustic ecology
- **Citizen Science**: Mobile app integration for species identification

## 👨‍💻 Author

**Joshua Payapulli**
- GitHub: [@Payapulli](https://github.com/Payapulli)
- USC ID: 3751786221

## 📚 References

- Anuran Calls (MFCCs) Dataset
- Scikit-learn Documentation: SVM and Multi-label Classification
- Bioacoustic Analysis: MFCC Feature Extraction
- Multi-label Learning: Evaluation Metrics and Strategies
