# Multimodal-dispute-resolution-engine
Smart AI microservice that analyzes disputes using LLM reasoning, EXIF forensics, and vision models to generate structured, confidence-based recommendations.

🧠 Overview

The Multimodal Dispute AI Engine is a stateless FastAPI microservice designed to bring structured intelligence to real-world disputes.

It combines:

🧠 Large Language Models (Mistral)

👁️ Vision-Language Models (LLaVA)

🔍 EXIF metadata forensics

⚖️ Evidence consistency analysis

to produce neutral, explainable, and confidence-based verdict recommendations.

✨ Why This Exists

Today, most dispute systems suffer from:

❌ manual review bottlenecks

❌ unverifiable evidence

❌ subjective decision-making

❌ fraud-prone workflows

This engine introduces a programmable trust layer that transforms messy disputes into structured, analyzable signals.

🔥 Key Features
🧠 LLM-Powered Case Reasoning

Argument extraction from both parties

Credibility scoring

Neutral case summarization

Structured recommendation generation

👁️ Vision Intelligence (LLaVA)

Scene understanding

Damage detection

Object recognition

Safety hazard identification

Indoor/outdoor classification

🔍 EXIF Forensic Analysis

Timestamp validation

Device metadata extraction

GPS presence checks

Software tamper signals

File fingerprinting

⚖️ Multimodal Evidence Fusion

The engine cross-validates:

Text claims

Visual evidence

Metadata signals

to produce:

Evidence strength score

Consistency score

Fraud indicators

Trust level classification

🏗️ Stateless Microservice Design

No database dependency

Horizontally scalable

Deterministic JSON outputs

Easy backend integration

Local-first privacy model

🧩 Architecture
Client → FastAPI → LLM (Mistral)
                  → Vision Model (LLaVA)
                  → EXIF Analyzer
                  → Evidence Fusion
                  → Structured Verdict JSON
🛠️ Tech Stack

Backend

FastAPI

Python 3.11+

Pydantic v2

AI Runtime

Ollama

Mistral 7B

LLaVA

Image Processing

Pillow

EXIF extraction

Infra

Uvicorn

Local GPU acceleration (optional)

📡 API
Primary Endpoint
POST /api/v1/analyze-case
📥 Example Request
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
📤 Example Output
{
  "case_summary": "...",
  "dao_recommendation": {
    "tenant_favor_confidence": 72,
    "landlord_favor_confidence": 15,
    "neutral_confidence": 13,
    "recommended_outcome": "Favor Tenant"
  },
  "vision_analyses": [...],
  "request_id": "..."
}
🚀 Getting Started
1️⃣ Prerequisites

Python 3.11+

Ollama installed

Mistral model pulled

LLaVA model pulled

2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Start Ollama
ollama serve
4️⃣ Run the API
uvicorn app.main:app --reload --port 8000
5️⃣ Open Docs
http://localhost:8000/docs
🧪 Performance

Tested locally on:

RTX 4050 (6GB VRAM)

16GB RAM

Local Ollama runtime

Typical latency:

⏱️ Text-only: ~3–5s

⏱️ With vision: ~6–10s

🧭 Example Use Cases

This engine is domain-agnostic and can power:

🏠 Rental dispute resolution

🛡️ Insurance claim verification

🛒 Marketplace fraud detection

📦 Warranty claim validation

🎧 Customer support arbitration

⚖️ DAO-assisted moderation

🔒 Design Principles

Privacy-first local inference

Deterministic structured outputs

Human-in-the-loop compatible

Fraud-aware evidence scoring

Modular multimodal pipeline

🚧 Current Limitations

Not a legal decision system

Requires publicly accessible images

Vision accuracy depends on image quality

Optimized for English text

🔮 Future Improvements

Batch inference pipeline

Streaming responses

Advanced tamper detection

Juror explanation mode

Lightweight edge deployment

👨‍💻 Author

<your-name>

Built as part of advanced work in:

Multimodal AI systems

Trust infrastructure

Automated dispute intelligence

AI-assisted governance

⭐ If This Project Helped You

Give it a star — it helps more than you think.