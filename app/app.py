import numpy as np
import pandas as pd
import gradio as gr
import spaces
from transformers import pipeline
from sentence_transformers import SentenceTransformer

STAR_MODEL = "Ronohcr7/distilbert-yelp-stars"
EMO_MODEL = "Ronohcr7/distilbert-emotion"
EMO_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

stars_clf = pipeline("text-classification", model=STAR_MODEL, top_k=None,
                     truncation=True, max_length=256)
emo_clf = pipeline("text-classification", model=EMO_MODEL, top_k=None,
                   truncation=True, max_length=128)
st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
emb = np.load("review_embeddings (1).npy")
reviews = pd.read_csv("reviews.csv")


@spaces.GPU
def analyze(text):
    if not text or not text.strip():
        return {}, {}
    s = stars_clf(text)[0]
    stars = {f"{int(r['label'].split('_')[-1]) + 1} star": float(r["score"]) for r in s}
    e = emo_clf(text)[0]
    emo = {EMO_LABELS[int(r["label"].split("_")[-1])]: float(r["score"]) for r in e}
    return stars, emo



def search(query, max_stars):
    if not query or not query.strip():
        return []
    try:
        q = st.encode([query], normalize_embeddings=True)
        scores = (emb @ q.T).ravel()
        scores = np.where(reviews["stars"].values <= int(max_stars), scores, -1)
        rows = []
        for j in scores.argsort()[::-1][:5]:
            rows.append([int(reviews["stars"].iloc[j]), round(float(scores[j]), 2),
                         str(reviews["text"].iloc[j])])
        return rows
    except Exception as e:
        return [[0, 0.0, f"Search error: {type(e).__name__}: {e}"]]


with gr.Blocks(title="Review Intelligence Platform") as demo:
    gr.Markdown(
        "# Review Intelligence Platform\n"
        "Paste a customer review to predict its star rating (fine-tuned DistilBERT, "
        "macro F1 0.61) and its emotion, or search 5,000 Yelp reviews by meaning."
    )
    with gr.Tab("Analyze a review"):
        txt = gr.Textbox(lines=5, label="Paste a review")
        btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            out_stars = gr.Label(num_top_classes=5, label="Predicted rating")
            out_emo = gr.Label(num_top_classes=6, label="Predicted emotion")
        btn.click(analyze, txt, [out_stars, out_emo])
        gr.Examples(
            examples=[
                ["The food was cold and the waiter ignored us for 30 minutes. Never coming back."],
                ["Absolutely loved this place! Friendly staff, amazing pasta, and fast service."],
                ["It was okay. Nothing special, but nothing terrible either."],
            ],
            inputs=txt,
        )
    with gr.Tab("Search reviews"):
        q = gr.Textbox(label="Describe what you are looking for",
                       placeholder="e.g. rude staff and long waits")
        ms = gr.Slider(1, 5, value=5, step=1, label="Show reviews up to this many stars")
        sbtn = gr.Button("Search", variant="primary")
        res = gr.Dataframe(headers=["stars", "similarity", "review"], wrap=True)
        sbtn.click(search, [q, ms], res)

demo.launch()
