# 🔍 DeepVerify
**AI-Generated Content Detection Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Sprint%201-yellow.svg)]()
[![Review](https://img.shields.io/badge/Review-In%20Progress-green.svg)]()

## 📌 Overview
DeepVerify is an open-source web platform that enables users to detect AI-generated text using state-of-the-art detection algorithms combined with explainable AI.

## 🎯 Problem Statement
AI-generated content is proliferating across academic, journalistic, and social contexts with no accessible verification tools for ordinary users. Current solutions (GPTZero, Turnitin) are expensive, closed-source, and enterprise-focused.

## 💡 Solution
- **Detection Engine**: Combines GLTR statistical analysis + Ghostbuster ML classification
- **Explainable AI**: Ollama/Mistral 7B generates plain-English explanations of detection reasoning
- **Accessible Interface**: Clean React UI requiring no technical expertise
- **Privacy-First**: All processing happens locally or on our servers — no third-party APIs

## 🏗️ System Architecture
```
┌─────────────────┐
│  React Frontend │ (Vite + TailwindCSS)
└────────┬────────┘
         │ HTTP/REST
┌────────▼────────┐
│   FastAPI       │ (Rate limiting + CORS)
└────────┬────────┘
         │
┌────────▼────────────────────┐
│  Detection Engine           │
│  • GLTR (HuggingFace GPT-2) │
│  • Ghostbuster (UMD 2023)   │
│  • Composite Scorer         │
└────────┬────────────────────┘
         │
┌────────▼─────────────────┐
│  Ollama + Mistral 7B     │ (Local inference)
└──────────────────────────┘
         │
┌────────▼────────┐
│   PostgreSQL    │ (Result storage)
└─────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18 + Vite + TailwindCSS | User interface |
| Backend | Python 3.11 + FastAPI | API server |
| Detection | Ghostbuster + GLTR | AI text analysis |
| AI Explain | Ollama + Mistral 7B | Plain-English explanations |
| Database | PostgreSQL | Result persistence |
| Deploy | Vercel + Render | Cloud hosting (free tier) |

## 👥 Team
- **[Your Name]** — Product Owner + Frontend Developer
- **[Partner Name]** — Scrum Master + Backend Developer

## 📅 Sprint Timeline

### Sprint 1 (Weeks 1-2): Foundation ✅ IN PROGRESS
- [x] Documentation and architecture
- [x] Product backlog creation
- [ ] React UI implementation
- [ ] GLTR integration
- [ ] FastAPI endpoint setup

### Sprint 2 (Weeks 3-4): Core Detection
- [ ] Ghostbuster integration
- [ ] Composite scoring algorithm
- [ ] Ollama/Mistral setup
- [ ] AI explanation generation

### Sprint 3 (Weeks 5-6): Features
- [ ] Sentence-level highlighting
- [ ] PDF report export
- [ ] File upload support
- [ ] Dockerization

### Sprint 4 (Weeks 7-8): Deployment
- [ ] Cloud deployment
- [ ] Rate limiting
- [ ] Shareable links
- [ ] Final polish

## 🎓 Academic Context
- **Course**: 21CSP302L — Third Year Project
- **Institution**: SRM Institute of Science and Technology
- **Review 1**: February 28, 2026
- **Review 2**: March 22, 2026
- **Final Review**: April 28-30, 2026

## 🌍 SDG Alignment
**Primary**: SDG 16 — Peace, Justice & Strong Institutions
- Protecting citizens from AI-generated misinformation
- Enabling informed democratic participation
- Defending academic integrity

**Secondary**: SDG 4 (Quality Education), SDG 9 (Innovation & Infrastructure)

## 📊 Project Status
- ✅ Planning phase complete
- ✅ Architecture finalized
- 🔄 Sprint 1 in progress
- ⏳ MVP target: Review 2 (March 22)

## 📄 License
MIT License — Fully open source

## 📞 Contact
- **Guide**: [Faculty Name]
- **Email**: [Your Email]
- **GitHub**: https://github.com/Anirdq/

---

*Built with ❤️ for digital trust and transparency*
```

Commit:
```
git add README.md
git commit -m "Add comprehensive project documentation"
git push origin main
