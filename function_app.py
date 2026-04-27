import azure.functions as func
import json
import logging
from recommender import get_recommendations, load_data

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Chargement des données au démarrage (une seule fois, pas à chaque requête)
load_data()


@app.route(route="recommendations")
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP trigger pour les recommandations d'articles.

    Méthodes acceptées : GET, POST

    GET  /api/recommend?user_id=12345&n=5
    POST /api/recommend  body: {"user_id": 12345, "n": 5}

    Réponse 200 :
        {
            "user_id": 12345,
            "recommendations": [101, 202, 303, 404, 505],
            "cold_start": false
        }
    """
    logging.info("Requête de recommandation reçue.")

    # ── Lecture des paramètres (GET ou POST) ──
    user_id = req.params.get("user_id")
    n = req.params.get("n", "5")

    if not user_id:
        try:
            body = req.get_json()
            user_id = body.get("user_id")
            n = body.get("n", 5)
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Paramètre 'user_id' manquant."}),
                status_code=400,
                mimetype="application/json",
            )

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "Paramètre 'user_id' manquant."}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        user_id = int(user_id)
        n = int(n)
    except (ValueError, TypeError):
        return func.HttpResponse(
            json.dumps({"error": "'user_id' et 'n' doivent être des entiers."}),
            status_code=400,
            mimetype="application/json",
        )

    # ── Génération des recommandations ──
    try:
        recommendations, is_cold_start = get_recommendations(user_id, n=n)
    except Exception as e:
        logging.exception("Erreur lors du calcul des recommandations.")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )

    response = {
        "user_id": user_id,
        "recommendations": recommendations,
        "cold_start": is_cold_start,
    }

    return func.HttpResponse(
        json.dumps(response),
        status_code=200,
        mimetype="application/json",
    )