# Fake News Detection: From TF-IDF Baseline to Fine-Tuned DistilBERT

This project classifies news articles as real or fake using two approaches:

* A TF-IDF + Logistic Regression baseline

* A fine-tuned DistilBERT transformer model

## Setup Instructions
1. Clone Repository
```
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

2. Create and Activate Virtual Environment (Optional)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install Dependencies

```
pip install -r requirements.txt
```
For DistilBERT training

```
pip install transformers datasets scikit-learn
```
4. Download Dataset
Get the WELFake dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and extract it to the data/ directory.


## Running the Models
TF-IDF + Logistic Regression
```
python 01_eda_and_baseline.py
```

This will:

* Clean the dataset

* Vectorize it using TF-IDF

* Train a Logistic Regression classifier

* Print evaluation metrics (accuracy ~95%)

Fine-Tuning DistilBERT (GPU recommended)

Open and run the Jupyter notebook:

```
jupyter notebook 02_transformer_finetuning.ipynb
```
Or run in Google Colab:

Upload 02_transformer_finetuning.ipynb

Enable GPU under Runtime > Change runtime type

The notebook will:

* Tokenize and encode the dataset

* Fine-tune distilbert-base-uncased with:

* Batch size: 16

* Epochs: 3

* Learning rate: 2e-5

Output a classification report and confusion matrix

