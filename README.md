# 🧠 Multimodal Dispute AI Engine

A stateless FastAPI microservice that brings structured intelligence to real-world disputes by combining large language models, vision-language models, EXIF metadata forensics, and evidence consistency analysis to produce **neutral, explainable, and confidence-based verdict recommendations**.

---

## ✨ Why This Exists

Most dispute systems suffer from:

- ❌ Manual review bottlenecks
- ❌ Unverifiable evidence
- ❌ Subjective decision-making
- ❌ Fraud-prone workflows

This engine introduces a **programmable trust layer** that transforms messy disputes into structured, analyzable signals.

---

## 🔥 Key Features

### 🧠 LLM-Powered Case Reasoning
- Argument extraction from both parties
- Credibility scoring
- Neutral case summarization
- Structured recommendation generation

### 👁️ Vision Intelligence (LLaVA)
- Scene understanding
- Damage detection
- Object recognition
- Safety hazard identification
- Indoor/outdoor classification

### 🔍 EXIF Forensic Analysis
- Timestamp validation
- Device metadata extraction
- GPS presence checks
- Software tamper signals
- File fingerprinting

### ⚖️ Multimodal Evidence Fusion

The engine cross-validates text claims, visual evidence, and metadata signals to produce:

- Evidence strength score
- Consistency score
- Fraud indicators
- Trust level classification

### 🏗️ Stateless Microservice Design
- No database dependency
- Horizontally scalable
- Deterministic JSON outputs
- Easy backend integration
- Local-first privacy model

---

## 🧩 Architecture

```
Client → FastAPI → LLM (Mistral)
                 → Vision Model (LLaVA)
                 → EXIF Analyzer
                 → Evidence Fusion
                 → Structured Verdict JSON
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, Python 3.11+, Pydantic v2 |
| **AI Runtime** | Ollama, Mistral 7B, LLaVA |
| **Image Processing** | Pillow, EXIF extraction |
| **Infra** | Uvicorn, Local GPU acceleration (optional) |

---

## 📡 API

### Primary Endpoint

```
POST /api/v1/analyze-case
```

### 📥 Example Request

```json
{
  "issue_id": "issue-abc123",
  "tenant_complaint": "Heating system broken for 3 weeks.",
  "landlord_response": "Repair scheduled.",
  "incident_date": "2024-01-10T00:00:00Z",
  "tenant_evidence": [
    {
      "file_url": "https://example.com/image.jpg",
      "exif_data": {
        "datetime_original": "2024:01:15 14:30:00"
      },
      "uploaded_at": "2024-01-20T10:30:00Z"
    }
  ],
  "landlord_evidence": [],
  "property_history": {
    "previous_complaints": 0,
    "resolution_rate": 0
  },
  "enable_vision_analysis": true
}
```

### 📤 Example Response

```json
{
  "case_summary": "...",
  "dao_recommendation": {
    "tenant_favor_confidence": 72,
    "landlord_favor_confidence": 15,
    "neutral_confidence": 13,
    "recommended_outcome": "Favor Tenant"
  },
  "vision_analyses": ["..."],
  "request_id": "..."
}
```

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed
- Mistral model pulled
- LLaVA model pulled

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start Ollama

```bash
ollama serve
```

### 4️⃣ Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 5️⃣ Open Interactive Docs

```
http://localhost:8000/docs
```

---

## 🧪 Performance

Tested locally on an RTX 4050 (6GB VRAM) with 16GB RAM using the local Ollama runtime.

| Mode | Typical Latency |
|------|----------------|
| ⏱️ Text-only | ~3–5s |
| ⏱️ With vision | ~6–10s |

---

## 🧭 Example Use Cases

This engine is domain-agnostic and can power:

| Domain | Application |
|--------|-------------|
| 🏠 Real Estate | Rental dispute resolution |
| 🛡️ Insurance | Claim verification |
| 🛒 E-Commerce | Marketplace fraud detection |
| 📦 Retail | Warranty claim validation |
| 🎧 Support | Customer support arbitration |
| ⚖️ Web3 | DAO-assisted moderation |

---

## 🔒 Design Principles

- **Privacy-first** — local inference keeps data off third-party servers
- **Deterministic outputs** — structured JSON for reliable downstream integration
- **Human-in-the-loop compatible** — recommendations, not mandates
- **Fraud-aware** — evidence scoring built with adversarial inputs in mind
- **Modular pipeline** — swap or extend any component independently

---

## 🚧 Current Limitations

- Not a legal decision system
- Requires publicly accessible image URLs
- Vision accuracy depends on image quality
- Optimized for English text

---

## 🔮 Future Improvements

- [ ] Batch inference pipeline
- [ ] Streaming responses
- [ ] Advanced tamper detection
- [ ] Juror explanation mode
- [ ] Lightweight edge deployment

---

## 👨‍💻 Author

Kaushike Ramanathan

Built as part of advanced work in multimodal AI systems, trust infrastructure, automated dispute intelligence, and AI-assisted governance.

---

## ⭐ Support This Project

If this project helped you, give it a star — it helps more than you think.
