import azure.functions as func
import json
import logging
import traceback
import time
from recommender import get_recommendations, get_item_based_recommendations, load_data

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Chargement des données au démarrage (une seule fois, pas à chaque requête)
try:
    start_time = time.time()
    load_data()
    elapsed = time.time() - start_time
    logging.info(f"✓✓✓ Données de recommandation chargées en {elapsed:.2f}s")
except Exception as e:
    logging.error(f"✗✗✗ ERREUR CRITIQUE au démarrage: {str(e)}")
    traceback.print_exc()


@app.route(route="recommendations")
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP trigger pour les recommandations d'articles.

    GET  /api/recommendations?user_id=12345&n=5
    POST /api/recommendations  body: {"user_id": 12345, "n": 5}

    Optimisations implémentées:
    - FAISS indexing pour recherche O(log N) au lieu de O(N)
    - float32 embeddings (-50% RAM vs float64)
    - Aucun full matrix multiplication en mémoire
    """
    start_time = time.time()
    
    # ── Lecture des paramètres (GET ou POST) ──
    user_id = req.params.get("user_id")
    n = req.params.get("n", "5")

    # ── Essayer de parser le JSON ──
    if not user_id:
        try:
            body = req.get_json()
            user_id = body.get("user_id")
            n = body.get("n", n)
        except ValueError as e:
            logging.warning(f"JSON invalide: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Body JSON invalide ou 'user_id' manquant."}),
                status_code=400,
                mimetype="application/json",
            )

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "Paramètre 'user_id' est obligatoire."}),
            status_code=400,
            mimetype="application/json",
        )

    # ── Validation des types ──
    try:
        user_id = int(user_id)
        n = int(n)
        
        if user_id < 0:
            raise ValueError("user_id ne peut pas être négatif")
        if n <= 0 or n > 100:
            raise ValueError("n doit être entre 1 et 100")
            
    except (ValueError, TypeError) as e:
        logging.warning(f"Paramètres invalides: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Paramètres invalides: {str(e)}"}),
            status_code=400,
            mimetype="application/json",
        )

    # ── Générer les recommandations ──
    try:
        recommendations, is_cold_start = get_recommendations(user_id, n=n)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        response = {
            "user_id": user_id,
            "recommendations": recommendations,
            "cold_start": is_cold_start,
            "count": len(recommendations),
            "response_time_ms": round(elapsed_ms, 2),
        }

        logging.info(
            f"✓ Recommandations: user_id={user_id}, count={len(recommendations)}, "
            f"cold_start={is_cold_start}, time={elapsed_ms:.0f}ms"
        )

        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json",
        )

    except RuntimeError as e:
        logging.error(f"✗ Erreur runtime: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Service non disponible. Les données ne sont pas chargées."}),
            status_code=503,
            mimetype="application/json",
        )
    
    except Exception as e:
        logging.exception(f"✗ Erreur pour user_id={user_id}")
        return func.HttpResponse(
            json.dumps({
                "error": "Erreur interne du serveur",
                "details": str(e)
            }),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="recommendations_item_based")
def recommend_item_based(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP trigger pour les recommandations ITEM-BASED.
    
    Recommande des articles similaires à ceux consultés par l'utilisateur.
    (Collaborative Filtering basé sur la similarité d'articles)

    GET  /api/recommendations_item_based?user_id=12345&n=5
    POST /api/recommendations_item_based  body: {"user_id": 12345, "n": 5}

    Approche:
    - Récupère l'historique de l'utilisateur (articles consultés)
    - Trouve les articles similaires via FAISS (similarité d'embeddings)
    - Combine les scores de similarité
    - Exclut les articles déjà vus
    """
    start_time = time.time()
    
    # ── Lecture des paramètres (GET ou POST) ──
    user_id = req.params.get("user_id")
    n = req.params.get("n", "5")

    # ── Essayer de parser le JSON ──
    if not user_id:
        try:
            body = req.get_json()
            user_id = body.get("user_id")
            n = body.get("n", n)
        except ValueError as e:
            logging.warning(f"JSON invalide: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Body JSON invalide ou 'user_id' manquant."}),
                status_code=400,
                mimetype="application/json",
            )

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "Paramètre 'user_id' est obligatoire."}),
            status_code=400,
            mimetype="application/json",
        )

    # ── Validation des types ──
    try:
        user_id = int(user_id)
        n = int(n)
        
        if user_id < 0:
            raise ValueError("user_id ne peut pas être négatif")
        if n <= 0 or n > 100:
            raise ValueError("n doit être entre 1 et 100")
            
    except (ValueError, TypeError) as e:
        logging.warning(f"Paramètres invalides: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Paramètres invalides: {str(e)}"}),
            status_code=400,
            mimetype="application/json",
        )

    # ── Générer les recommandations ITEM-BASED ──
    try:
        print(f"DEBUG: Appel de get_item_based_recommendations avec user_id={user_id}, n={n}")
        recommendations, is_cold_start = get_item_based_recommendations(user_id, n=n)
        print("10")
        elapsed_ms = (time.time() - start_time) * 1000
        
        response = {
            "user_id": user_id,
            "method": "item-based-collaborative-filtering",
            "recommendations": recommendations,
            "cold_start": is_cold_start,
            "count": len(recommendations),
            "response_time_ms": round(elapsed_ms, 2),
        }
        print("20")
        logging.info(
            f"✓ Recommandations Item-Based: user_id={user_id}, count={len(recommendations)}, "
            f"cold_start={is_cold_start}, time={elapsed_ms:.0f}ms"
        )
        print("30")
        print(f"Response: {response}")
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json",
        )

    except RuntimeError as e:
        logging.error(f"✗ Erreur runtime: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Service non disponible. Les données ne sont pas chargées."}),
            status_code=503,
            mimetype="application/json",
        )
    
    except Exception as e:
        logging.exception(f"✗ Erreur Item-Based pour user_id={user_id}")
        return func.HttpResponse(
            json.dumps({
                "error": "Erreur interne du serveur",
                "details": str(e)
            }),
            status_code=500,
            mimetype="application/json",
        )