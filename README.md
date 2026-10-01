# Review Intelligence Platform

An end-to-end NLP project that reads customer reviews and predicts star ratings, built phase by phase from classical machine learning to neural networks and transformers. Each phase is measured against the previous one so it is clear what deep learning adds.

**Dataset:** [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full), a 20,000-review sample for training and 5,000 for testing. The five star ratings are evenly balanced.

**Metric:** macro F1 (accuracy is also reported). Random guessing would score about 20%.

## Results (test set)

| Phase | Model | Accuracy | Macro F1 |
|---|---|---|---|
| 4 | Fine-tuned DistilBERT | 0.6078 | **0.6084** |
| 1 | TF-IDF + Logistic Regression | 0.5632 | 0.5604 |
| 2 | TF-IDF + MLP (PyTorch) | 0.5378 | 0.5366 |
| 3 | BiLSTM + embeddings | 0.5376 | 0.5313 |
| 1 | TF-IDF + XGBoost | 0.5202 | 0.5160 |
| 1 | TF-IDF + Random Forest | 0.5010 | 0.4879 |

## Key findings

- **Linear models are hard to beat on sparse text.** Logistic regression outperformed Random Forest, XGBoost and a neural network on TF-IDF features.
- **The bottleneck was the representation, not model capacity.** TF-IDF throws away word order, so a bigger model on the same input did not help.
- **Learning embeddings from scratch is limited by data.** A BiLSTM reached 0.531 macro F1 and overfitted after epoch 3, because 18k reviews is too few to learn word meanings from scratch.
- **Pretrained transformers win.** Fine-tuned DistilBERT reached 0.608 macro F1, about 5 points above the best classical model, using the same 20k training reviews.

## Error analysis (DistilBERT)

Almost all errors are between neighbouring ratings: 88% of mistakes (1,734 of 1,961) are off by exactly one star, and the model is within one star on 95.5% of reviews. The extreme ratings are easiest (F1 0.74 for 1 star, 0.69 for 5 stars), while 2, 3 and 4 stars score 0.52 to 0.56. The largest confusions are 4 stars predicted as 5 stars (289 reviews) and 1 star predicted as 2 stars (242). Many of these reviews are genuinely ambiguous, which sets a ceiling for any model on this task.

**Possible improvements:** train on more data (this run used 20k of 650k reviews), try a larger model such as RoBERTa, or treat the rating as an ordinal target.

## Roadmap

- [x] Phase 1: classical ML baselines
- [x] Phase 2: neural network (MLP) in PyTorch
- [x] Phase 3: LSTM with word embeddings
- [x] Phase 4: fine-tuned DistilBERT
- [ ] Phase 5: emotion detection on reviews (reusing [my emotion classifier](https://huggingface.co/spaces/Ronohcr7/emotion-classifier))
- [ ] Phase 6: topic modeling and summarization
- [ ] Phase 7: semantic search
- [ ] Phase 8: dashboard deployed on Hugging Face Spaces

## Tech stack

Python, scikit-learn, XGBoost, PyTorch, Hugging Face Transformers and Datasets

## Notebooks

- `01_classical_ml.ipynb`: TF-IDF with Logistic Regression, Random Forest, XGBoost
- `02_neural_network.ipynb`: MLP built and trained in PyTorch
- `03_lstm.ipynb`: BiLSTM with learned embeddings
- `04_distilbert.ipynb`: fine-tuned DistilBERT and error analysis
