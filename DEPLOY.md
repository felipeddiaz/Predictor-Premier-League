# Deployment Guide — Premier League Predictor

## Architecture

```
Netlify (React frontend)
        |
        v
Render free tier (FastAPI backend)
        |
        v
ML Model (.pkl files loaded at startup)
```

## 1. Backend — Deploy to Render (Free Tier)

### Option A: Blueprint (recommended)
1. Push this repo to GitHub
2. Go to [render.com](https://render.com) > New > Blueprint Instance
3. Connect your GitHub repo — Render reads `render.yaml` automatically
4. Click "Apply" — it will build and deploy the API

### Option B: Manual
1. Go to [render.com](https://render.com) > New > Web Service
2. Connect your GitHub repo
3. Settings:
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile.api`
   - **Plan**: Free
   - **Health Check Path**: `/health`
4. Click "Create Web Service"

### Verify
Once deployed, visit: `https://your-service.onrender.com/docs`

You should see the Swagger UI with all endpoints.

## 2. Frontend — Deploy to Netlify

1. Go to [netlify.com](https://netlify.com) > Add new site > Import from Git
2. Connect your GitHub repo
3. Settings:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
4. **Environment variable** (important!):
   - `VITE_API_URL` = `https://your-render-service.onrender.com`
5. Click "Deploy site"

### Update API URL
After deploying the backend, update `frontend/.env.production`:
```
VITE_API_URL=https://your-actual-render-url.onrender.com
```

## 3. Local Development

```bash
# Backend only
pip install -r requirements.txt
pip install -r requeriments-api.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Frontend only
cd frontend
npm install
npm run dev

# Both with Docker
docker compose up
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/teams` | List available teams |
| `POST` | `/predict` | Simple prediction (team names only) |
| `POST` | `/predictions` | Detailed prediction (with odds) |
| `POST` | `/predictions/detail` | Full prediction with binary markets |
| `GET` | `/docs` | Swagger UI |

### Quick Test
```bash
curl -X POST https://your-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'
```

## Free Tier Limitations

### Render Free Tier
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30-60 seconds (cold start)
- 750 hours/month free

### Netlify Free Tier
- 100 GB bandwidth/month
- 300 build minutes/month
- Unlimited sites

## Future: Pre-loaded Predictions

To add daily auto-predictions:

1. Create a `scripts/generate_predictions.py` that runs all upcoming fixtures
2. Store results in a JSON file or simple database (SQLite/Supabase free tier)
3. Add a `GET /predictions/upcoming` endpoint
4. Use a cron job (GitHub Actions schedule or Render cron) to refresh daily

Example GitHub Actions cron:
```yaml
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
```
