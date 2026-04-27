import pickle
import logging

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
load_dotenv(override=True)

# ── Variables globales (chargées une seule fois au démarrage de l'instance) ──
user_vectors = None
user_seen = None
embeddings = None
cold_start = None
articles = None

from azure.storage.blob import BlobServiceClient
import pickle
import pandas as pd
import io

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
container_name = "data"

blob_service = BlobServiceClient.from_connection_string(connection_string)

def load_pickle(blob_name):
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    data = blob_client.download_blob().readall()
    return pickle.loads(data)

def load_csv(blob_name):
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    data = blob_client.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))



def load_data(data_dir: str = "data") -> None:
    """
    Charge les données pré-calculées en mémoire.
    Doit être appelée une seule fois au démarrage (dans function_app.py).
    """
    global user_vectors, user_seen, embeddings, cold_start, articles

    # logging.info("Chargement des données de recommandation…")

    # with open(f"{data_dir}/user_vectors.pickle", "rb") as f:
    #     user_vectors = pickle.load(f)
    
    # with open(f"{data_dir}/user_seen.pickle", "rb") as f:
    #     user_seen = pickle.load(f)

    # with open(f"{data_dir}/articles_embeddings.pickle", "rb") as f:
    #     embeddings = pickle.load(f)

    # with open(f"{data_dir}/cold_start.pickle", "rb") as f:
    #     cold_start = pickle.load(f)

    # articles = pd.read_csv(f"{data_dir}/articles_metadata.csv")
    
    
    user_vectors = load_pickle("user_vectors.pickle")
    user_seen = load_pickle("user_seen.pickle")
    embeddings = load_pickle("articles_embeddings.pickle")
    cold_start = load_pickle("cold_start.pickle")
    articles = load_csv("articles_metadata.csv")
    
    
    logging.info("Données chargées avec succès.")


def get_recommendations(user_id: int, n: int = 5) -> tuple[list, bool]:
    """
    Retourne les n articles recommandés pour un utilisateur.

    Returns:
        (article_ids, is_cold_start)
    """
    if user_vectors is None:
        raise RuntimeError("Les données ne sont pas chargées. Appelez load_data() d'abord.")

    # ── Cold start ──
    if user_id not in user_vectors:
        logging.warning(f"[COLD START] Utilisateur {user_id} inconnu → recommandations populaires")
        return cold_start[:n], True

    # ── Récupérer le vecteur utilisateur ──
    user_vector = user_vectors[user_id]          # shape: (250,)
    seen_articles = set(user_seen[user_id])

    # ── Similarité cosinus avec tous les articles ──
    scores = cosine_similarity([user_vector], embeddings)[0]  # shape: (N,)

    # ── DataFrame article_id → score ──
    results = pd.DataFrame({
        "article_id": articles["article_id"].values,
        "score": scores[articles["article_id"].values],
    })

    # ── Exclure les articles déjà vus ──
    results = results[~results["article_id"].isin(seen_articles)]

    # ── Top N ──
    top_n = results.sort_values("score", ascending=False).head(n)

    return top_n["article_id"].tolist(), False



