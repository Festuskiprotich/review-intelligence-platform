# Review Intelligence Platform

An end-to-end NLP project that reads customer reviews and predicts star ratings, built phase by phase from classical machine learning to neural networks and transformers. Each phase is measured against the previous one so it is clear what deep learning adds.

**Live demo:** https://huggingface.co/spaces/Ronohcr7/review-intelligence

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

## Emotion analysis (Phase 5)

I ran my fine-tuned [emotion classifier](https://huggingface.co/spaces/Ronohcr7/emotion-classifier) on 5,000 Yelp test reviews to see how predicted emotion relates to star rating. The share of reviews classified as joy rises from 26% at 1 star to 82% at 5 stars, while anger falls from 46% to 6% and sadness from 22% to 4%. Love, fear and surprise stay between 1% and 5% at every rating and carry little signal.

**Caveats:** the emotion model was trained on short tweets, while these are long business reviews, so this is an exploratory analysis. No ground-truth emotion labels exist for Yelp, so accuracy cannot be measured. About a quarter of 1-star reviews were still labelled joy, which I have not yet explained.

## Topic analysis and summarization (Phase 6)

I embedded 5,000 Yelp reviews with a sentence-transformer (all-MiniLM-L6-v2) and grouped them into 8 clusters with KMeans. The clusters mostly separate business types: general dining, Asian restaurants, hotels and casinos, coffee and desserts, casual dining, and stores and personal care. Two clusters stand out for low ratings. One about restaurant service and waiting averages 2.06 stars (493 of its 702 reviews are rated 1-2 stars), and one about customer service disputes averages 1.78 stars (388 of 479). Every other cluster averages between 2.95 and 3.62.

To see what the unhappy customers say, I summarized the five 1-2 star reviews closest to each problem cluster's centre with DistilBART. In the restaurant cluster they describe mediocre or cold food, dirty tables and long waits. In the other cluster they describe missed deliveries, unresponsive staff and an auto shop upselling tires.

**Caveats:** the number of clusters (8) was chosen without tuning, the cluster names are my interpretation of keywords and example reviews, and the clusters describe business type more than complaint type. Each summary covers only five reviews, so the findings are illustrations and not measurements, and the summarizer sometimes picks a positive sentence from a negative review.

## Semantic search (Phase 7)

I embedded 5,000 Yelp reviews with all-MiniLM-L6-v2 and ranked them by cosine similarity to a free-text query. The search matches meaning, not keywords, and it picked up sentiment without being told. The query "the waiter ignored us and the food arrived cold" returned five 1-2 star reviews, mostly about long waits, rude service and cold food, and "friendly and fast service" returned five 4-5 star reviews praising friendly staff and good service. A rating filter narrows results further: "rude staff and billing problems" restricted to 2 stars or fewer returned five 1-star reviews about rude staff and poor customer service.

**Caveats:** I checked this by reading the top five results for four queries, not against labelled relevance judgements. Vague queries were weaker: "great place for a quiet date night" returned two relevant reviews followed by three 3-star reviews that matched only loosely, and the billing part of the third query was not clearly matched. Search covers only the 5,000-review sample.

## Live dashboard (Phase 8)

**Try it:** https://huggingface.co/spaces/Ronohcr7/review-intelligence

The dashboard has two tabs. *Analyze a review* predicts the star rating with a fine-tuned DistilBERT and the emotion with my emotion classifier. *Search reviews* finds the most similar reviews to a free-text query among 5,000 Yelp reviews, with an optional rating filter. The app code is in the `app/` folder.

**Note:** the dashboard uses a retrained copy of the Phase 4 model with identical settings. It scored 0.612 accuracy and 0.6126 macro F1 on the same test set, within normal run-to-run variation of the 0.6084 reported above. The app runs on Hugging Face's free ZeroGPU hardware, so the first request after a period of inactivity can take several seconds.
The free ZeroGPU hardware is occasionally unavailable, in which case the Analyze tab shows an error and works again after a short wait.

## Roadmap

- [x] Phase 1: classical ML baselines
- [x] Phase 2: neural network (MLP) in PyTorch
- [x] Phase 3: LSTM with word embeddings
- [x] Phase 4: fine-tuned DistilBERT
- [x] Phase 5: emotion analysis of reviews
- [x] Phase 6: topic modeling and summarization
- [x] Phase 7: semantic search
- [x] Phase 8: dashboard deployed on Hugging Face Spaces

## Tech stack

Python, scikit-learn, XGBoost, PyTorch, Hugging Face Transformers and Datasets, sentence-transformers, Gradio, Hugging Face Spaces, matplotlib

## Notebooks

- `01_classical_ml.ipynb`: TF-IDF with Logistic Regression, Random Forest, XGBoost
- `02_neural_network.ipynb`: MLP built and trained in PyTorch
- `03_lstm.ipynb`: BiLSTM with learned embeddings
- `04_distilbert.ipynb`: fine-tuned DistilBERT and error analysis
- `05_emotions.ipynb`: emotion analysis of reviews
- `06_topics.ipynb`: topic clustering and summarization
- `07_semantic_search.ipynb`: semantic search over reviews
- `08_dashboard_prep.ipynb`: retrains the star-rating model and prepares the dashboard files

## App

- `app/app.py`: the Gradio dashboard
- `app/requirements.txt`: its dependencies
