# Toxic Comment Classification using NLP and Deep Learning

A Python project that loads a dataset of online comments, cleans and analyzes the text, and trains a deep learning model to classify whether a comment is toxic. The project covers the full pipeline: data exploration, text preprocessing, tokenization, model training, and performance evaluation.

## Project Overview

Toxic comments (threats, identity attacks, severe toxicity) harm online communities. This project builds an NLP pipeline that processes raw comment text and trains a neural network to detect toxicity automatically, which can support content moderation.

## Dataset

- File: `toxic_comments_dataset.csv`
- Main text column: `Comment_Text`
- Target column: `Toxicity_Label`
- Additional columns include `Severe_Toxic`, `Identity_Attack`, `Threat`, `Toxicity_Score`, `Comment_Length`, `Word_Count`, `Platform`, `Topic`, `Moderation_Priority`, and `Language`.

## Workflow

| Task | Description |
|------|-------------|
| 1 | Dataset loading and initial exploration |
| 2 | Text cleaning, preprocessing, data quality checks, and target analysis |
| 3 | NLP tokenization and sequence padding |
| 4 | Train and validate the deep learning model |
| 5 | Evaluate, plot, and visualize model performance |

### Text cleaning steps
- Convert text to lowercase
- Remove special characters, punctuation, and numbers (keep letters only)
- Remove extra whitespace
- Create a new `Clean_Comment` column

## Tech Stack

- Python
- Pandas
- Matplotlib and Seaborn
- TensorFlow / Keras (Tokenizer, pad_sequences, Sequential model)
- NumPy and scikit-learn (label encoding, evaluation)
- Jupyter Notebook

## Model Architecture

Sequential Keras model:

`Embedding -> LSTM -> Dropout -> Dense -> Dropout -> Dense (softmax)`

- Optimizer: Adam
- Metric: accuracy
- Epochs: 15, batch size: 32, validation split: 20%

## Project Structure

```
toxic-comments-analysis-python/
├── solution2.ipynb              # Full notebook with all 5 tasks
├── toxic_comments_dataset.csv   # Dataset
└── README.md
```

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/shahidlaiba899-ops/toxic-comments-analysis-python.git
   cd toxic-comments-analysis-python
   ```
2. Install the required libraries:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
   ```
3. Open `solution2.ipynb` in Jupyter Notebook or VS Code and run all cells.

## Results

- Test accuracy: 63.15% (test loss: 1.07)
- Accuracy and loss curves, a confusion matrix, and sample predictions on new comments are shown in Task 5 of the notebook.

**Limitations:** the dataset is imbalanced (most comments are labeled Clean), so the current model mostly predicts the majority class. Planned improvements: class weights, dataset balancing, a Bidirectional LSTM/GRU, and precision/recall/F1 reporting.

## Author

**Laiba Shahid**
GitHub: [shahidlaiba899-ops](https://github.com/shahidlaiba899-ops)
