# RouteIQ: A Multi-Agent ML System for Intelligent Customer Support Routing

RouteIQ is a multi-agent ML pipeline that automates customer support ticket triage. For every incoming ticket, it delivers three outputs: a **queue routing decision with a confidence score**, **contextual retrieval of similar resolved tickets (RAG)**, and an **LLM-generated draft reply** ready for agent review.

---

## System Architecture

Built on **LangGraph** with 8 agent nodes and a shared `AgentState` store.

**Offline Training Phase** (runs once)

| Node | Role |
|------|------|
| 1. preprocess-data | Load CSV, clean text, fit TF-IDF vectoriser, stratified 80/10/10 split |
| 2. train-models | Fit Logistic Regression + Random Forest with GridSearchCV |
| 3. evaluate-models | Per-class Precision / Recall / F1, Macro F1, confusion matrix |
| 4. select-model | Compare Macro F1, store best model with written rationale |

**Online Inference Phase** (runs per ticket)

| Node | Role |
|------|------|
| 5. classify-complaint | TF-IDF transform → predict queue + confidence score |
| 6. retrieve-similar-tickets | RAG: cosine similarity over training corpus, return top-3 |
| 7. lookup-order-record | Mock CRM lookup: customer name, plan, contract value |
| 8. draft-response | GPT-4o-mini: routing recommendation + draft customer reply |

Tickets below **50% confidence** are automatically flagged for human review.

---

## Results

| Model | Macro F1 | Weighted F1 | Accuracy |
|-------|----------|-------------|----------|
| Random Forest | **0.527** | 0.537 | 53.8% |
| Logistic Regression | 0.511 | 0.509 | 50.8% |
| Baseline (majority class) | 0.045 | 0.132 | 29.3% |

Strongest queues: Service Outages & Maintenance (F1=0.684), Technical Support (F1=0.608).

---

## Tech Stack

Python · LangGraph · scikit-learn (TF-IDF, Random Forest, Logistic Regression) · GPT-4o-mini · RAG

Dataset: [Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) — 28,580 tickets (EN/DE)

---

## Known Failure Modes & Proposed Fix

**Technical cluster confusion**: Product Support, IT Support, and Technical Support share overlapping vocabulary, causing 400+ misrouted tickets within this cluster.

**Semantically hollow tickets**: High-level business language without domain-specific terms causes misrouting (e.g. "strategies for brand growth" routed to Customer Service instead of Product Support).

**Proposed fix**: Replace TF-IDF with `paraphrase-multilingual-MiniLM-L12-v2` to capture semantic similarity across paraphrases and language variants.
