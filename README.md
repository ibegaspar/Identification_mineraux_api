# API Prédiction Minéraux - Team RobotMali

API FastAPI pour identifier les minéraux à partir d'images et propriétés physiques.

## 🚀 Déploiement avec Docker

### Prérequis
- Docker installé
- Docker Compose installé

### Déploiement local

1. **Construire l'image Docker :**
```bash
docker build -t api-mineraux .
```

2. **Lancer avec Docker Compose :**
```bash
docker-compose up -d
```

3. **Ou lancer directement :**
```bash
docker run -p 8000:8000 -v $(pwd):/app api-mineraux
```

### Vérification
- API accessible sur : http://localhost:8000
- Documentation Swagger : http://localhost:8000/docs
- Health check : http://localhost:8000/health

## 📋 Endpoints

### POST /predict
Prédiction complète avec image + propriétés physiques
- `image`: Fichier image (JPG, PNG)
- `durete`: Dureté sur l'échelle de Mohs (0-10)
- `densite`: Densité en g/cm³ (0-20)

### POST /predict_simple
Prédiction basée uniquement sur les propriétés physiques
- `durete`: Dureté sur l'échelle de Mohs (0-10)
- `densite`: Densité en g/cm³ (0-20)

### GET /health
Vérification de l'état de l'API

## 🐳 Déploiement sur plateformes cloud

### Render.com
1. Connecter votre repo GitHub
2. Créer un nouveau "Web Service"
3. Sélectionner le repo
4. Configuration :
   - Build Command : `docker build -t api-mineraux .`
   - Start Command : `docker run -p $PORT:8000 api-mineraux`

### Railway.app
1. Connecter votre repo GitHub
2. Créer un nouveau projet
3. Railway détectera automatiquement le Dockerfile

### Fly.io
1. Installer flyctl
2. `fly launch`
3. Suivre les instructions

## 📁 Structure des fichiers
```
├── api.py                 # Code principal de l'API
├── requirements.txt       # Dépendances Python
├── Dockerfile            # Configuration Docker
├── docker-compose.yml    # Configuration Docker Compose
├── .dockerignore         # Fichiers exclus du build
├── model_final_durete_densiter.h5  # Modèle Keras
├── scaler.pkl           # Scaler scikit-learn
├── label_encoder.pkl    # Encodeur de labels
└── data_test.csv        # Données de test
```

## 🔧 Variables d'environnement
- `PORT`: Port d'écoute (défaut: 8000)
- `PYTHONUNBUFFERED=1`: Pour les logs en temps réel

## 📝 Notes importantes
- Les modèles doivent être présents dans le répertoire de l'application
- L'API accepte les images JPG et PNG
- Les limites de validation : dureté (0-10), densité (0-20)
- CORS configuré pour permettre les requêtes depuis n'importe quelle origine 