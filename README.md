# hybrid-quantum-self-attention-

# Hybrid Quantum Self-Attention for Small-Scale Language Modeling

A research dashboard that implements and benchmarks a **hybrid quantum self-attention layer**, comparing it against classical scaled dot-product self-attention. 

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20Simulation-792EE5)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tensors-EE4C2C?logo=pytorch&logoColor=white)](#)

> Replace the first badge link / add your deployed Streamlit Cloud URL once you've deployed it (see [Deployment](#deployment) below).

---

## Project Objective

Design and simulate a hybrid quantum self-attention layer for small-scale language modeling, and quantify its **latency overhead** relative to classical self-attention, using a noiseless quantum simulator (PennyLane `default.qubit`) so the study is fully reproducible on classical hardware.

## Problem Statement

Classical self-attention scales as **O(n²·d)** in sequence length `n` and model dimension `d`, which becomes a bottleneck for longer contexts. Quantum computing offers, in principle, exponentially large Hilbert spaces from a small number of qubits. This project investigates whether a small (4-qubit) variational quantum circuit can be embedded inside a self-attention block to enrich attention scores with entanglement-based correlations — and measures the practical latency cost of doing so on a simulator, as a necessary first step before any real-hardware deployment.

## How It Works

```
Input Text → Tokenizer → Embedding Layer → Hybrid Quantum Self-Attention → Feed-Forward → Output (Next Token)
```

- **Classical path:** standard scaled dot-product self-attention, `Attention(Q,K,V) = softmax(QKᵗ / √d_k) V`.
- **Quantum path:** the same Q/K/V projections are computed classically, but each query vector's leading 4 dimensions are additionally encoded as **RY rotation angles** into a 4-qubit circuit. A ring of **CNOT gates** entangles the qubits, and the resulting **Pauli-Z expectation values** modulate that query's attention scores before the softmax.

## Dashboard Pages

| Page | Contents |
|---|---|
| **Overview** | Objective, problem statement, architecture diagram, explanation of hybrid quantum self-attention |
| **NLP Fundamentals** | Tokenization demo, embedding visualization, sequence-length vs. compute-cost chart, language modeling primer |
| **Next Word Prediction** | Toy bigram model with classical vs. quantum-reranked predictions |
| **Attention Visualization** | Side-by-side classical vs. hybrid quantum attention heatmaps + difference map |
| **Quantum Circuit Explorer** | Interactive 4-qubit circuit with adjustable RY angles, circuit diagram, live measurement outputs |
| **Performance Analysis** | Latency benchmarking across sequence lengths (64 / 128 / 256) with Plotly charts and tables |
| **Research Findings** | Key observations, limitations, future work |

## Tech Stack

- **Streamlit** — UI / dashboard framework
- **PyTorch** — tensor operations, classical attention math
- **PennyLane** — 4-qubit quantum circuit simulation (`default.qubit` device)
- **NumPy / Pandas** — numerical computing and tabular data
- **Matplotlib** — attention heatmaps, circuit diagrams, architecture diagram
- **Plotly** — interactive charts (latency, sequence-length comparisons)

## Project Structure

```
quantum-language-dashboard/
├── app.py              # Full Streamlit application (all pages)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Running Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/hybrid-quantum-self-attention.git
cd hybrid-quantum-self-attention
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deployment

This app is deployable for free on **Streamlit Community Cloud**:

1. Push this repository to GitHub (done).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select this repo → set main file to `app.py` → **Deploy**.
4. Streamlit Cloud installs `requirements.txt` automatically (first build can take several minutes due to PyTorch) and gives you a public URL.

## Research Notes / Limitations

- All experiments run on a **noiseless classical simulator** (PennyLane `default.qubit`); no real quantum hardware noise or shot noise is modeled.
- The next-word prediction demo uses a small toy bigram corpus, not a trained transformer — it illustrates the quantum re-ranking concept, not real language-modeling accuracy.
- The quantum circuit is invoked once per token per forward pass, so latency overhead grows with sequence length on a classical simulator; this scaling would differ on real quantum hardware or with batched circuit execution.

See the **Research Findings** page in the app for the full list of observations, limitations, and proposed future work.

## License

MIT — feel free to fork and extend for your own research.
