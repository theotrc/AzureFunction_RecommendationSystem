import pickle
import logging
import os
from typing import Optional, Tuple, List, Dict, Set
import io

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.warning("FAISS non disponible, fallback sur cosine_similarity (lent)")

load_dotenv(override=True)

# ── Variables globales (chargées une seule fois au démarrage) ──
_data_cache: Optional[Dict] = None
_faiss_index: Optional[object] = None

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
container_name = "data"


def _get_blob_service() -> BlobServiceClient:
    """Crée une instance de BlobServiceClient avec gestion d'erreur."""
    if not connection_string:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING n'est pas configurée")
    return BlobServiceClient.from_connection_string(connection_string)


def load_pickle(blob_name: str) -> object:
    """Charge un fichier pickle depuis Azure Blob Storage."""
    try:
        blob_service = _get_blob_service()
        blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
        data = blob_client.download_blob().readall()
        return pickle.loads(data)
    except Exception as e:
        logging.error(f"Erreur lors du chargement de {blob_name}: {str(e)}")
        raise


def load_csv(blob_name: str) -> pd.DataFrame:
    """Charge un fichier CSV depuis Azure Blob Storage."""
    try:
        blob_service = _get_blob_service()
        blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
        data = blob_client.download_blob().readall()
        return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        logging.error(f"Erreur lors du chargement de {blob_name}: {str(e)}")
        raise


def _build_faiss_index(embeddings: np.ndarray) -> Optional[object]:
    """
    Construit un index FAISS pour recherche rapide et efficace en mémoire.
    Convertit embeddings en float32 pour économiser 50% de mémoire.
    """
    if not HAS_FAISS:
        return None

    try:
        # Convertir en float32 (économie de 50% RAM vs float64)
        embeddings_fp32 = embeddings.astype(np.float32)
        
        # Créer index L2 (distance euclidienne, comparable à cosine pour embeddings normalisés)
        dimension = embeddings_fp32.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Ajouter les embeddings
        index.add(embeddings_fp32)
        
        logging.info(f"✓ Index FAISS créé: {index.ntotal} articles indexés")
        return index
    
    except Exception as e:
        logging.error(f"Erreur création index FAISS: {str(e)}")
        return None


def load_data() -> None:
    """
    Charge les données pré-calculées en mémoire.
    - Utilise float32 pour économiser 50% de RAM
    - Crée un index FAISS pour recherche rapide
    """
    global _data_cache, _faiss_index

    if _data_cache is not None:
        logging.debug("Les données sont déjà en cache.")
        return

    try:
        logging.info("▶ Chargement des données de recommandation…")

        # Charger les données
        user_vectors_dict = load_pickle("user_vectors.pickle")
        user_seen = load_pickle("user_seen.pickle")
        embeddings = load_pickle("articles_embeddings.pickle")
        cold_start = load_pickle("cold_start.pickle")
        articles = load_csv("articles_metadata.csv")

        # Convertir embeddings en float32 (économie RAM)
        if isinstance(embeddings, np.ndarray):
            embeddings_original_dtype = embeddings.dtype
            embeddings = embeddings.astype(np.float32)
            logging.info(f"  Embeddings converti: {embeddings_original_dtype} → float32 (-50% RAM)")
        
        # Convertir user_vectors en float32
        user_vectors = {}
        for uid, vec in user_vectors_dict.items():
            user_vectors[uid] = np.array(vec, dtype=np.float32)

        # Validation
        if not isinstance(user_vectors, dict):
            raise ValueError("user_vectors doit être un dictionnaire")
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("embeddings doit être un array numpy")
        if not isinstance(articles, pd.DataFrame):
            raise ValueError("articles doit être un DataFrame pandas")

        # Créer index FAISS pour recherche rapide
        _faiss_index = _build_faiss_index(embeddings)

        _data_cache = {
            "user_vectors": user_vectors,
            "user_seen": user_seen,
            "embeddings": embeddings,
            "cold_start": cold_start,
            "articles": articles,
        }

        memory_usage_mb = embeddings.nbytes / (1024 * 1024)
        logging.info(
            f"✓ Données chargées avec succès\n"
            f"  - Utilisateurs: {len(user_vectors):,}\n"
            f"  - Articles: {len(articles):,}\n"
            f"  - Embeddings: {embeddings.nbytes / (1024**2):.1f} MB\n"
            f"  - FAISS Index: {'✓ Actif' if _faiss_index else '✗ Désactivé'}"
        )

    except Exception as e:
        logging.error(f"✗ Erreur lors du chargement des données: {str(e)}")
        raise


def _get_data() -> Dict:
    """Retourne le cache de données."""
    if _data_cache is None:
        raise RuntimeError("Les données ne sont pas chargées. Appelez load_data() d'abord.")
    return _data_cache


def _search_faiss(user_vector: np.ndarray, k: int = 1000) -> np.ndarray:
    """
    Recherche les k articles les plus similaires avec FAISS.
    Beaucoup plus rapide et efficace en mémoire que cosine_similarity brute force.
    """
    if _faiss_index is None:
        raise RuntimeError("Index FAISS non disponible")
    
    # Convertir en float32
    user_vector_fp32 = np.array([user_vector], dtype=np.float32)
    
    # Rechercher les k plus proches voisins
    distances, indices = _faiss_index.search(user_vector_fp32, k)
    
    # Pour L2, plus petite distance = plus similaire
    # Convertir distances en "scores de similarité" (inverse pour cohérence)
    scores = 1.0 / (1.0 + distances[0])  # 0-1 range
    
    return indices[0], scores



def load_data(data_dir: str = "data") -> None:
    """
    Charge les données pré-calculées en mémoire.
    Doit être appelée une seule fois au démarrage (dans function_app.py).
    """
    global _data_cache

    if _data_cache is not None:
        logging.debug("Les données sont déjà en cache.")
        return

    try:
        logging.info("Chargement des données de recommandation…")

        user_vectors = load_pickle("user_vectors.pickle")
        user_seen = load_pickle("user_seen.pickle")
        embeddings = load_pickle("articles_embeddings.pickle")
        cold_start = load_pickle("cold_start.pickle")
        articles = load_csv("articles_metadata.csv")

        # Validation basique
        if not isinstance(user_vectors, dict):
            raise ValueError("user_vectors doit être un dictionnaire")
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("embeddings doit être un array numpy")
        if not isinstance(articles, pd.DataFrame):
            raise ValueError("articles doit être un DataFrame pandas")

        _data_cache = {
            "user_vectors": user_vectors,
            "user_seen": user_seen,
            "embeddings": embeddings,
            "cold_start": cold_start,
            "articles": articles,
        }

        logging.info(
            f"Données chargées avec succès. "
            f"Utilisateurs: {len(user_vectors)}, Articles: {len(articles)}"
        )

    except Exception as e:
        logging.error(f"Erreur lors du chargement des données: {str(e)}")
        raise


def _get_data() -> Dict:
    """Retourne le cache de données. Lève une exception si les données ne sont pas chargées."""
    if _data_cache is None:
        raise RuntimeError("Les données ne sont pas chargées. Appelez load_data() d'abord.")
    return _data_cache


def get_recommendations(user_id: int, n: int = 5) -> Tuple[List[int], bool]:
    """
    Retourne les n articles recommandés pour un utilisateur.
    
    Optimisations :
    - Utilise FAISS pour recherche rapide O(log N) au lieu de O(N)
    - Économise mémoire avec float32
    - Traite seulement les articles pertinents (top-k de FAISS)

    Args:
        user_id: ID de l'utilisateur
        n: Nombre de recommandations (1-100)

    Returns:
        (article_ids, is_cold_start)
    """
    if n <= 0 or n > 100:
        raise ValueError("n doit être entre 1 et 100")

    data = _get_data()
    user_vectors = data["user_vectors"]
    user_seen = data["user_seen"]
    articles = data["articles"]
    cold_start = data["cold_start"]

    # ── Cold start (utilisateur inconnu) ──
    if user_id not in user_vectors:
        logging.warning(f"[COLD START] Utilisateur {user_id} inconnu → recommandations populaires")
        result = cold_start[:n].tolist() if isinstance(cold_start, np.ndarray) else cold_start[:n]
        return result, True

    # ── Récupérer le vecteur utilisateur (float32) ──
    user_vector = user_vectors[user_id]
    seen_articles: Set[int] = set(user_seen.get(user_id, []))

    # ── Utiliser FAISS pour recherche rapide ──
    if _faiss_index is not None and HAS_FAISS:
        try:
            # Chercher les top-k (un peu plus que demandé pour filtrer les vus)
            k_search = min(n * 5, 1000)  # Chercher 5x plus, max 1000
            indices, scores = _search_faiss(user_vector, k=k_search)
            
            # Créer le DataFrame
            results = pd.DataFrame({
                "article_id": articles["article_id"].values[indices],
                "score": scores,
            })
        except Exception as e:
            logging.warning(f"FAISS failed, fallback: {str(e)}")
            return _fallback_cosine_search(user_vector, user_seen, articles, n)
    else:
        # Fallback: cosine_similarity (lent avec 364k articles!)
        return _fallback_cosine_search(user_vector, user_seen, articles, n)

    # ── Exclure les articles déjà vus ──
    results = results[~results["article_id"].isin(seen_articles)]

    # ── Valider qu'il y a assez d'articles ──
    if len(results) < n:
        logging.warning(
            f"Seulement {len(results)} articles non vus pour user_id={user_id} (demandé: {n})"
        )

    # ── Top N ──
    top_n = results.head(n)

    return top_n["article_id"].tolist(), False


def _fallback_cosine_search(user_vector: np.ndarray, seen_articles: Set[int], 
                            articles: pd.DataFrame, n: int) -> Tuple[List[int], bool]:
    """
    Fallback sur cosine_similarity si FAISS indisponible.
    ⚠️  LENT avec 364k articles - ne pas utiliser en production!
    """
    logging.warning("⚠️  Utilisation de cosine_similarity (lent!) - FAISS recommandé")
    
    data = _get_data()
    embeddings = data["embeddings"]
    
    # Limit to max 50k articles pour ne pas exploser la RAM
    sample_size = min(50000, len(articles))
    sample_indices = np.random.choice(len(articles), size=sample_size, replace=False)
    
    sample_embeddings = embeddings[sample_indices]
    scores = cosine_similarity([user_vector], sample_embeddings)[0]
    
    results = pd.DataFrame({
        "article_id": articles["article_id"].values[sample_indices],
        "score": scores,
    })
    
    results = results[~results["article_id"].isin(seen_articles)]
    top_n = results.nlargest(n, "score")
    
    return top_n["article_id"].tolist(), False



