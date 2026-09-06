# Bachelorarbeit: Finetuning eines BGE-Modells für Fact-Checking

Dieses Repository enthält den vollständigen Code und die Evaluation für die Bachelorarbeit zum Thema **Finetuning von Embedding-Modellen für automatisierte Fact-Checking-Pipelines**.

Das Hauptziel ist die Optimierung eines Dense Retrievers (**`BAAI/bge-base-en-v1.5`**) mittels Hard-Negative Mining und Matryoshka-Loss, kombiniert mit einem lokalen LLM (**`Qwen/Qwen2.5-7B-Instruct`**) für ein RAG-basiertes Fact-Checking System.

## Architektur & Modellbasis

- **Basis Retrieval-Modell:** [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) *(Feinabgestimmt via Matryoshka Loss & Hard-Negative Mining)*
- **Generator-Modell:** [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) *(Lokale Ausführung über Hugging Face Transformers Pipeline)*
- **Fakten-Datenbank:** [FEVER (Fact Extraction and VERification Dataset)](https://fever.ai/dataset/fever.html)

---

## Features & Pipeline

* **Hard-Negative Mining:** Dynamische Generierung von herausfordernden Negativ-Beispielen (`util.mine_hard_negatives`).
* **Matryoshka Loss Training:** Vorbereitet auf flexible Vektordimensionen (`[64, 128, 256, 512, 768]`).
* **Dense Retrieval Evaluation:** Messung von `Accuracy@k` (S@1, S@3, S@5, S@10) und `MRR` gegen das Baseline-Modell.
* **End-to-End RAG Inference:** Lokale Klassifikation von Behauptungen (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) inkl. Evidenz-Scoring.

---
## Schnellstart & Ausführung
Das Notebook ist für die direkte Ausführung in Google Colab (idealerweise mit T4/A100 GPU) optimiert.

1. Öffne das Notebook Fact-Checking.ipynb in Google Colab.
2. Stelle sicher, dass die GPU-Laufzeit aktiviert ist (Laufzeit -> Laufzeittyp ändern -> GPU).
3. Führe die ersten Setup-Zellen aus, um die nötigen Bibliotheken zu installieren.
4. Die Dateien in dem Data-Ordner inkl. des FEVER-Datensatzes müssen im base_path vorhanden sein.
---


