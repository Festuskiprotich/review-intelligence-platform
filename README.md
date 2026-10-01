# Review Intelligence Platform

An end-to-end NLP project that reads customer reviews and predicts star ratings, built phase by phase from classical machine learning to neural networks and transformers. Each phase is measured against the previous one so it is clear what deep learning adds.

**Dataset:** [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full), a 20,000-review sample for training and 5,000 for testing. The five star ratings are evenly balanced, about 4,000 reviews each in the training sample.

**Metric:** macro F1 (accuracy is also reported). Random guessing would score about 20%.

## Results (test set)

| Phase | Model | Accuracy | Macro F1 |
|---|---|---|---|
| 1 | TF-IDF + Logistic Regression | 0.5632 | **0.5604** |
| 2 | TF-IDF + MLP (PyTorch) | 0.5378 | 0.5366 |
| 3 | BiLSTM + embeddings | 0.5376 | 0.5313 |
| 1 | TF-IDF + XGBoost | 0.5202 | 0.5160 |
| 1 | TF-IDF + Random Forest | 0.5010 | 0.4879 |
| 4 | Fine-tuned DistilBERT | planned | planned |
## Key findings

- **Linear models are hard to beat on sparse text.** Logistic regression outperformed Random Forest, XGBoost and a neural network on TF-IDF features.
- **The bottleneck is the representation, not model capacity.** TF-IDF discards word order and context, so a bigger model on the same input did not help. The MLP (0.537) stayed below logistic regression (0.560).
- **Next:** models that read words in order (LSTM) and pretrained transformers (DistilBERT) should capture the nuance between neighbouring ratings such as 2 and 3 stars.

## Roadmap

- [x] Phase 1: classical ML baselines
- [x] Phase 2: neural network (MLP) in PyTorch
- [ ] Phase 3: LSTM with word embeddings
- [ ] Phase 4: fine-tuned DistilBERT
- [ ] Phase 5: emotion detection on reviews (reusing [my emotion classifier](https://huggingface.co/spaces/Ronohcr7/emotion-classifier))
- [ ] Phase 6: topic modeling and summarization
- [ ] Phase 7: semantic search
- [ ] Phase 8: dashboard deployed on Hugging Face Spaces

## Tech stack

Python, scikit-learn, XGBoost, PyTorch, Hugging Face Transformers and Datasets

## Notebooks

- `01_classical_ml.ipynb`: TF-IDF with Logistic Regression, Random Forest, XGBoost
- `02_neural_network.ipynb`: MLP built and trained in PyTorch
