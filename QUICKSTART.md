# DeepVerify – Quick Start Scripts
# ─────────────────────────────────

# 1. Start the FastAPI backend
# Windows PowerShell:
#   cd backend
#   python -m venv venv
#   .\venv\Scripts\Activate.ps1
#   pip install -r requirements.txt
#   uvicorn main:app --reload --port 8000

# 2. Start the Vite frontend (separate terminal)
#   cd frontend
#   npm install
#   npm run dev

# Both should now be running:
#   Frontend: http://localhost:5173
#   Backend:  http://localhost:8000
#   API docs: http://localhost:8000/docs

# Sprint 2 additions (not yet):
#   ollama pull mistral:7b
#   # Then update backend to hit http://localhost:11434/api/generate
