# Deployment Instructions for SFFM Live App

## 🚀 Quick Start

### Local Deployment

#### 1. **Using Streamlit Cloud (Recommended - Free)**
```bash
# 1. Push your code to GitHub (already done!)
# 2. Visit https://share.streamlit.io
# 3. Sign in with GitHub
# 4. Select repo: farahqoonitaa/sffm-phd-research
# 5. Select main branch
# 6. Set main file path to: app.py
# 7. Click Deploy!

# Your app will be live at: https://sffm-phd-research.streamlit.app
```

#### 2. **Local Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at: http://localhost:8501
```

#### 3. **Manual Local Deployment**
```bash
# Clone the repo
git clone https://github.com/farahqoonitaa/sffm-phd-research.git
cd sffm-phd-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🌐 Production Deployment Options

### Option 1: Streamlit Cloud (Easiest, $0/month)
- **Pros**: Free, automatic deploys from GitHub, built-in SSL
- **Setup**: Just connect your GitHub repo
- **URL**: `https://sffm-phd-research.streamlit.app`

### Option 2: Heroku (Deprecated but available alternatives)
- **Alternative**: Railway.app ($5-20/month)
```bash
railway link
railway up
```

### Option 3: AWS Elastic Container Service (ECS)
```bash
# Build and push Docker image
docker build -t sffm-app:latest .
docker tag sffm-app:latest <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/sffm-app:latest
docker push <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/sffm-app:latest

# Deploy via CloudFormation/ECS
```

### Option 4: DigitalOcean App Platform ($5+/month)
```yaml
# app.yaml for DigitalOcean
name: sffm-app
services:
  - name: streamlit
    github:
      repo: farahqoonitaa/sffm-phd-research
      branch: main
    build_command: pip install -r requirements.txt
    run_command: streamlit run app.py --server.port=8080
    http_port: 8080
```

### Option 5: Hugging Face Spaces (Free)
- Visit https://huggingface.co/spaces
- Create new space with Streamlit SDK
- Connect GitHub repo

---

## 📋 Pre-Deployment Checklist

- [x] `requirements.txt` updated with Streamlit
- [x] `app.py` created and tested locally
- [x] `.streamlit/config.toml` configured
- [x] `Dockerfile` and `docker-compose.yml` ready
- [ ] Environment variables configured (if needed)
- [ ] GitHub repo is public

---

## 🔧 Environment Variables (Optional)

Create `.streamlit/secrets.toml` for sensitive data:
```toml
# Database credentials
DB_HOST = "your_db_host"
DB_USER = "your_db_user"
DB_PASSWORD = "your_db_password"

# API keys
OPENAI_API_KEY = "your_api_key"
```

---

## 📊 App Features

The live app includes 7 interactive pages:

1. **🏠 Home** - Project overview and research questions
2. **🔬 Gaussian Process Demo** - Interactive GP uncertainty visualization
3. **⚖️ Fairness Algorithms** - Compare 5 fairness approaches
4. **📊 Uncertainty Calibration** - Reliability diagrams and ECE metrics
5. **💳 Risk Pricing Simulator** - Interactive pricing with fairness constraints
6. **📈 Heterogeneity Analysis** - Western vs. informal-economy comparison
7. **📚 Documentation** - API reference, datasets, and resources

---

## 🚨 Troubleshooting

### Port Already in Use
```bash
# Find process on port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8502
```

### Memory Issues
```bash
# Streamlit runs in server mode by default
# For large models, consider:
streamlit run app.py --logger.level=warning --client.maxMessageSize=200
```

### CORS Issues (Cloud deployment)
```toml
# Add to .streamlit/config.toml
[client]
showErrorDetails = true

[server]
enableCORS = false
enableXsrfProtection = false
```

---

## 📈 Monitoring & Analytics

### Streamlit Cloud Built-in Metrics
- View analytics at: https://share.streamlit.io/admin
- Monitor usage, errors, and performance

### Custom Analytics (Optional)
Add Google Analytics or Mixpanel:
```python
import streamlit as st
st.set_page_config(
    page_title="SFFM",
    initial_sidebar_state="expanded"
)
# Analytics tracked automatically
```

---

## 🔄 Continuous Deployment

### GitHub Actions Auto-Deploy
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Streamlit Cloud
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Streamlit Cloud
        run: |
          # Streamlit Cloud auto-deploys on push
          # No additional action needed
          echo "App deployed!"
```

---

## 📞 Support

- **Issues**: Report on GitHub Issues
- **Email**: farahqoonita2@gmail.com
- **Documentation**: See README.md

---

**Last Updated**: June 2026
**Status**: Ready for Live Deployment ✅
