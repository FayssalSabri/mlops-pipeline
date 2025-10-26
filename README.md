# 🚀 MLOps Pipeline – Déploiement d’un modèle de Machine Learning

## 🎯 Objectif
Mettre en place un pipeline complet pour déployer un modèle de Machine Learning :
- Entraînement automatique (`train.py`)
- API via FastAPI
- Conteneurisation avec Docker
- CI/CD avec GitHub Actions

---

## ⚙️ Installation locale

```bash
git clone https://github.com/<ton_user>/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
