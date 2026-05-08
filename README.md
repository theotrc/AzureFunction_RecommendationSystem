# Application de Recommandation de Contenu

API Azure Functions pour générer des recommandations d'articles en utilisant des algorithmes de machine learning et collaborative filtering.

## 🚀 Routes disponibles

### 1. Recommandations Collaborative Filtering (User-Based)

**Endpoint:** `/api/recommendations`

**Description:** Recommande des articles basés sur les préférences de l'utilisateur en comparant son historique avec celui d'autres utilisateurs.

#### Paramètres d'entrée

| Paramètre | Type | Obligatoire | Description | Limites |
|-----------|------|-------------|-------------|---------|
| `user_id` | integer | ✅ | Identifiant unique de l'utilisateur | ≥ 0 |
| `n` | integer | ❌ | Nombre d'articles à recommander | 1-100 (défaut: 5) |

#### Formats de requête

**GET:**
```
GET /api/recommendations?user_id=12345&n=5
```

**POST:**
```
POST /api/recommendations
Content-Type: application/json

{
  "user_id": 12345,
  "n": 5
}
```

#### Réponse (200 OK)

```json
{
  "user_id": 12345,
  "recommendations": [
    {
      "article_id": 1,
      "title": "Titre de l'article",
      "score": 0.92
    },
    {
      "article_id": 2,
      "title": "Titre de l'article 2",
      "score": 0.87
    }
  ],
  "cold_start": false,
  "count": 2,
  "response_time_ms": 45.23
}
```

#### Erreurs possibles

| Code | Message | Raison |
|------|---------|--------|
| 400 | `user_id` est obligatoire | Paramètre manquant |
| 400 | Paramètres invalides | `user_id` négatif ou `n` hors limites (1-100) |
| 503 | Service non disponible | Données non chargées au démarrage |
| 500 | Erreur interne du serveur | Exception non prévue |

---

### 2. Recommandations Item-Based (Similarité d'articles)

**Endpoint:** `/api/recommendations_item_based`

**Description:** Recommande des articles similaires à ceux consultés par l'utilisateur en utilisant la similarité d'embeddings.

#### Paramètres d'entrée

| Paramètre | Type | Obligatoire | Description | Limites |
|-----------|------|-------------|-------------|---------|
| `user_id` | integer | ✅ | Identifiant unique de l'utilisateur | ≥ 0 |
| `n` | integer | ❌ | Nombre d'articles à recommander | 1-100 (défaut: 5) |

#### Formats de requête

**GET:**
```
GET /api/recommendations_item_based?user_id=12345&n=5
```

**POST:**
```
POST /api/recommendations_item_based
Content-Type: application/json

{
  "user_id": 12345,
  "n": 5
}
```

#### Réponse (200 OK)

```json
{
  "user_id": 12345,
  "method": "item-based-collaborative-filtering",
  "recommendations": [
    {
      "article_id": 5,
      "title": "Titre de l'article similaire",
      "score": 0.95
    },
    {
      "article_id": 8,
      "title": "Titre de l'article similaire 2",
      "score": 0.88
    }
  ],
  "cold_start": false,
  "count": 2,
  "response_time_ms": 52.15
}
```

#### Erreurs possibles

| Code | Message | Raison |
|------|---------|--------|
| 400 | `user_id` est obligatoire | Paramètre manquant |
| 400 | Paramètres invalides | `user_id` négatif ou `n` hors limites (1-100) |
| 503 | Service non disponible | Données non chargées au démarrage |
| 500 | Erreur interne du serveur | Exception non prévue |

---

## 📊 Structure de la réponse

### Champs communs

| Champ | Type | Description |
|-------|------|-------------|
| `user_id` | integer | ID de l'utilisateur demandeur |
| `recommendations` | array | Liste des articles recommandés |
| `cold_start` | boolean | Indique si l'utilisateur est nouveau (cold start) |
| `count` | integer | Nombre d'articles retournés |
| `response_time_ms` | float | Temps de traitement en millisecondes |

### Format d'un article recommandé

```json
{
  "article_id": 123,
  "title": "Titre de l'article",
  "score": 0.92
}
```

| Champ | Type | Description |
|-------|------|-------------|
| `article_id` | integer | ID unique de l'article |
| `title` | string | Titre de l'article |
| `score` | float | Score de recommandation (0.0 - 1.0) |

---

## 🔧 Optimisations implémentées

- **FAISS Indexing:** Recherche vectorielle optimisée en O(log N) au lieu de O(N)
- **Float32 Embeddings:** Réduction de 50% de la RAM vs float64
- **Cache des données:** Chargement unique au démarrage
- **Gestion des cold starts:** Détection automatique des utilisateurs nouveaux

---

## 📋 Exemples d'utilisation

### cURL

```bash
# User-Based Recommendations
curl -X GET "https://your-function-url/api/recommendations?user_id=12345&n=5"

# Item-Based Recommendations
curl -X POST "https://your-function-url/api/recommendations_item_based" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345, "n": 5}'
```

### Python

```python
import requests

# User-Based
response = requests.get(
    "https://your-function-url/api/recommendations",
    params={"user_id": 12345, "n": 5}
)
print(response.json())

# Item-Based
response = requests.post(
    "https://your-function-url/api/recommendations_item_based",
    json={"user_id": 12345, "n": 5}
)
print(response.json())
```

### JavaScript

```javascript
// User-Based
fetch('https://your-function-url/api/recommendations?user_id=12345&n=5')
  .then(r => r.json())
  .then(data => console.log(data));

// Item-Based
fetch('https://your-function-url/api/recommendations_item_based', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id: 12345, n: 5 })
})
  .then(r => r.json())
  .then(data => console.log(data));
```

---

## ⚙️ Configuration

Voir les fichiers de configuration:
- `requirements.txt` - Dépendances Python
- `local.settings.json` - Configuration locale
- `host.json` - Configuration Azure Functions

---

## 📝 Licence

MIT
