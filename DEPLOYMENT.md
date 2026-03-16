# Deployment Guide - PL Predictor

## 🚀 Opciones de Deployment

### 1. Docker Compose (Recomendado - Local & Dev)

```bash
# Iniciar servicios
docker-compose up

# Iniciar en background
docker-compose up -d

# Ver logs
docker-compose logs -f api
docker-compose logs -f frontend

# Detener servicios
docker-compose down

# Remover volúmenes
docker-compose down -v
```

**URLs:**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

### 2. Manual (Para Desarrollo)

#### Backend API
```bash
# Instalar dependencias
pip install -r requirements.txt
pip install -r requeriments-api.txt

# Variables de entorno
cp .env.example .env
# Editar .env según necesario

# Iniciar API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend
```bash
# Instalar dependencias
cd frontend
npm install

# Desarrollo
npm run dev        # http://localhost:3000

# Build para producción
npm run build

# Preview del build
npm run preview
```

---

### 3. Docker Individual

#### Compilar imagen API
```bash
docker build -f Dockerfile.api -t pl-predictor-api:latest .
```

#### Ejecutar contenedor API
```bash
docker run -p 8000:8000 \
  -v $(pwd)/api:/app/api \
  -v $(pwd)/core:/app/core \
  pl-predictor-api:latest
```

#### Compilar imagen Frontend
```bash
cd frontend
docker build -t pl-predictor-frontend:latest .
```

#### Ejecutar contenedor Frontend
```bash
docker run -p 3000:3000 \
  -e VITE_API_URL=http://localhost:8000/api \
  pl-predictor-frontend:latest
```

---

## 📋 Checklist Pre-Deploy

### ✅ Local Testing
- [ ] `pytest` - Todos los tests pasan
- [ ] `docker-compose up` - Servicios inician sin errores
- [ ] API Health: `curl http://localhost:8000/health`
- [ ] Frontend carga: `http://localhost:3000`
- [ ] Predicciones funcionan: `POST /predictions`

### ✅ Code Quality
- [ ] No hay imports no usados
- [ ] Código sigue PEP 8
- [ ] Docstrings presentes en funciones públicas
- [ ] Sin hardcoded passwords o secrets

### ✅ Security
- [ ] Validación de inputs (Pydantic)
- [ ] CORS configurado correctamente
- [ ] No hay vulnerabilidades conocidas: `pip audit`
- [ ] Variables sensibles en .env (no en código)

### ✅ Documentation
- [ ] README.md actualizado
- [ ] INTERFACE_SETUP.md completo
- [ ] TESTING.md con instrucciones
- [ ] Comentarios en código complejo

---

## 🌐 Deployment a Producción

### Opción A: Heroku

#### 1. Setup
```bash
heroku login
heroku create pl-predictor
```

#### 2. Archivo Procfile
```
web: python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

#### 3. Deploy
```bash
git push heroku main
heroku logs --tail
```

### Opción B: DigitalOcean App Platform

#### 1. Crear app.yaml
```yaml
name: pl-predictor
services:
  - name: api
    source:
      type: github
      repo: your-repo/Predictor-Premier-League
      branch: main
    dockerfile_path: Dockerfile.api
    http_port: 8000
    env:
      - key: API_PORT
        value: "8000"

  - name: frontend
    source:
      type: github
      repo: your-repo/Predictor-Premier-League
      branch: main
      source_dir: frontend
    dockerfile_path: Dockerfile
    http_port: 3000
```

#### 2. Deploy
```bash
doctl apps create --spec app.yaml
```

### Opción C: AWS ECS

```bash
# Crear repositorio ECR
aws ecr create-repository --repository-name pl-predictor

# Build y push
docker build -f Dockerfile.api -t pl-predictor-api .
docker tag pl-predictor-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/pl-predictor:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/pl-predictor:latest
```

### Opción D: Google Cloud Run

```bash
# Build
gcloud builds submit --tag gcr.io/PROJECT-ID/pl-predictor-api

# Deploy
gcloud run deploy pl-predictor-api \
  --image gcr.io/PROJECT-ID/pl-predictor-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🔐 Configuración de Seguridad

### Variables de Entorno Requeridas
```bash
# .env en producción
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false           # NUNCA true en producción
LOG_LEVEL=INFO
CORS_ORIGINS=["https://midominio.com"]  # Específico en prod
```

### HTTPS / SSL
```bash
# Con Nginx + Certbot
sudo apt-get install certbot python3-certbot-nginx
sudo certbot certonly --standalone -d midominio.com
```

### Reverse Proxy (Nginx)
```nginx
upstream api_backend {
    server api:8000;
}

server {
    listen 443 ssl http2;
    server_name midominio.com;

    ssl_certificate /etc/letsencrypt/live/midominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/midominio.com/privkey.pem;

    location /api {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://frontend:3000;
    }
}
```

---

## 📊 Monitoring & Logging

### Logs Aplicación
```bash
# Docker
docker-compose logs -f api

# Archivo local
tail -f logs/api.log
```

### Health Checks
```bash
# Endpoint health
curl http://localhost:8000/health

# Full status
curl http://localhost:8000/status
```

### Métricas (Opcional - Prometheus)
```bash
pip install prometheus-client
```

---

## 🔄 CI/CD Pipeline Ejemplo

### GitHub Actions (.github/workflows/deploy.yml)
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requeriments-api.txt
      - run: pytest

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -f Dockerfile.api -t pl-predictor:latest .
      - name: Deploy to server
        run: |
          # Script de deploy aquí
          ssh user@server "cd /app && docker-compose pull && docker-compose up -d"
```

---

## 🔧 Rollback

```bash
# Volver a versión anterior
docker-compose down
git checkout <commit-anterior>
docker-compose up -d

# O con Docker
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  watchtower --cleanup
```

---

## 📈 Performance Tuning

### Uvicorn Workers
```bash
# Producción: 4 workers
gunicorn -w 4 -b 0.0.0.0:8000 api.main:app

# O en docker-compose.yml
command: gunicorn -w 4 -b 0.0.0.0:8000 api.main:app
```

### Caching
```bash
# Redis cache (opcional)
pip install redis
# Configurar en api/main.py
```

---

## 📞 Troubleshooting

### Puerto en uso
```bash
# Encontrar proceso
lsof -i :8000
kill -9 <PID>
```

### Permisos Docker
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Logs de errores
```bash
docker-compose logs --tail=100 api
docker-compose ps  # Ver estado de servicios
```

---

## ✅ Post-Deployment Checks

1. API responde: `curl http://production-url/health`
2. Predicciones funcionan: `curl -X POST http://production-url/predictions`
3. Frontend carga: Visitar `http://production-url`
4. SSL válido (si aplica): `https://production-url`
5. Logs no muestran errores: `docker-compose logs`

---

## 📚 Recursos Adicionales

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Uvicorn](https://www.uvicorn.org/)
- [Nginx Configuration](https://nginx.org/en/docs/)
