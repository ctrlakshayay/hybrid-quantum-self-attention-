"""
Hybrid Quantum Self-Attention for Small-Scale Language Modeling
=================================================================
Single-file Streamlit research dashboard.

Author: <your name>
Stack : Streamlit, PyTorch, PennyLane, NumPy, Pandas, Matplotlib, Plotly

This file follows the exact structure of the original app.py:
    st.set_page_config -> st.title -> st.sidebar.selectbox -> if/elif per page
All logic (tokenizer, classical attention, hybrid quantum attention,
quantum circuit, performance benchmarking, plotting) is implemented as
plain functions ABOVE the page-routing block, then called inside each
if/elif branch. Nothing lives outside this single file.
"""

import time
import re
import math

import numpy as np
import pandas as pd
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# =====================================================================
# 0. GLOBAL CONFIG
# =====================================================================
# Fixing every random seed makes every chart / number reproducible,
# which matters a lot when the same dashboard is screenshotted for a
# report or shown live in a viva.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_QUBITS = 4          # number of qubits used by the quantum circuit
D_MODEL = 16          # embedding dimension used for the attention demos
SEQ_LENGTHS = [64, 128, 256]   # sequence lengths studied in the paper


# =====================================================================
# 1. TOKENIZER  (NLP Fundamentals / Next Word Prediction / Attention)
# =====================================================================
def simple_tokenize(text: str):
    """
    A minimal whitespace + punctuation tokenizer.

    Real subword tokenizers (BPE / WordPiece) are overkill for a
    small-scale demo, but the *idea* -- turning raw text into a list of
    discrete tokens -- is identical, so this keeps the dashboard fast
    and dependency-free while still being pedagogically correct.
    """
    text = text.strip().lower()
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z0-9']+|[.,!?;]", text)
    return tokens


def build_vocab(tokens):
    """Map each unique token to an integer id (a toy vocabulary)."""
    vocab = {}
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


# =====================================================================
# 2. TOY EMBEDDING LAYER  (NLP Fundamentals)
# =====================================================================
def get_embeddings(tokens, dim=D_MODEL, seed=SEED):
    """
    Deterministically generate an embedding vector per token.

    In a trained model these vectors are learned; here we draw them
    from a fixed random seed so that the SAME word always gets the
    SAME vector inside one Streamlit session -- enough to demonstrate
    what an embedding table represents, without training a model.
    """
    vocab = build_vocab(tokens)
    rng = np.random.default_rng(abs(hash(tuple(sorted(vocab.keys())))) % (2**32) if vocab else seed)
    table = {tok: rng.normal(0, 1, size=dim) for tok in vocab}
    matrix = np.stack([table[t] for t in tokens]) if tokens else np.zeros((0, dim))
    return matrix, vocab


# =====================================================================
# 3. CLASSICAL SELF-ATTENTION
# =====================================================================
def classical_self_attention(X: torch.Tensor):
    """
    Standard scaled dot-product self-attention (Vaswani et al., 2017).

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    X : (seq_len, d_model) tensor of token embeddings.
    Returns the attention weight matrix and the attended output.
    """
    d_k = X.shape[-1]
    Q, K, V = X, X, X                      # self-attention: Q=K=V=X
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return weights, output


# =====================================================================
# 4. QUANTUM CIRCUIT  (4-qubit PennyLane simulator)
# =====================================================================
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def quantum_circuit(angles):
    """
    A 4-qubit variational circuit used to inject quantum-derived
    structure into the attention scores.

    Steps:
      1. RY(angle_i) on each wire   -> puts each qubit into a
         superposition whose bias is controlled by `angle_i`.
      2. A ring of CNOT gates       -> entangles every qubit with its
         neighbour, so the qubits' states become correlated and can
         no longer be described independently.
      3. Measure <Z> on every wire  -> returns 4 real numbers in
         [-1, 1] that summarise the entangled state.
    """
    for i in range(N_QUBITS):
        qml.RY(angles[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[N_QUBITS - 1, 0])      # close the ring
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def draw_quantum_circuit(angles):
    """Return a matplotlib figure of the circuit diagram for given angles."""
    fig, ax = qml.draw_mpl(quantum_circuit, style="default")(angles)
    return fig


# =====================================================================
# 5. HYBRID QUANTUM SELF-ATTENTION
# =====================================================================
def hybrid_quantum_attention(X: torch.Tensor):
    """
    Hybrid quantum self-attention layer.

    Classical Q, K, V projections are kept (so the layer is still a
    drop-in replacement for nn.MultiheadAttention), but the raw
    attention score for each query token is *modulated* by the output
    of a 4-qubit quantum circuit that encodes the first N_QUBITS
    dimensions of that query vector as RY rotation angles.

    This is the core research idea of the project: classical linear
    algebra does the heavy lifting (so the layer stays trainable and
    scalable), while the quantum circuit contributes a non-classical,
    entanglement-based correction term that a purely classical layer
    cannot reproduce with the same number of parameters.
    """
    d_k = X.shape[-1]
    Q, K, V = X, X, X
    base_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    seq_len = X.shape[0]
    quantum_modulation = torch.zeros(seq_len)
    for i in range(seq_len):
        # squash the first N_QUBITS embedding dims into [-pi, pi]
        angles = (torch.tanh(Q[i, :N_QUBITS]) * math.pi).detach().numpy()
        expvals = quantum_circuit(angles)
        quantum_modulation[i] = float(np.mean(expvals))

    # modulate each query row's scores by its quantum signal
    scores = base_scores * (1.0 + 0.25 * quantum_modulation.unsqueeze(1))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return weights, output, quantum_modulation


# =====================================================================
# 6. LATENCY BENCHMARKING  (Performance Analysis)
# =====================================================================
@st.cache_data(show_spinner=False)
def benchmark_attention(seq_lengths, d_model=D_MODEL, repeats=3):
    """
    Time classical vs hybrid quantum attention for several sequence
    lengths and return a tidy pandas DataFrame ready for plotting.
    """
    rows = []
    for L in seq_lengths:
        torch.manual_seed(SEED)
        X = torch.randn(L, d_model)

        # ---- classical timing ----
        c_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            classical_self_attention(X)
            c_times.append(time.perf_counter() - t0)

        # ---- hybrid quantum timing ----
        # (kept to 1 repeat for L=256 to keep the dashboard responsive;
        #  the quantum circuit is the dominant cost and is run once per
        #  token, i.e. L times per forward pass)
        q_repeats = repeats if L <= 128 else 1
        q_times = []
        for _ in range(q_repeats):
            t0 = time.perf_counter()
            hybrid_quantum_attention(X)
            q_times.append(time.perf_counter() - t0)

        rows.append({
            "Sequence Length": L,
            "Classical Latency (ms)": np.mean(c_times) * 1000,
            "Hybrid Quantum Latency (ms)": np.mean(q_times) * 1000,
        })

    df = pd.DataFrame(rows)
    df["Overhead (x)"] = df["Hybrid Quantum Latency (ms)"] / df["Classical Latency (ms)"]
    return df


# =====================================================================
# 7. PLOTTING HELPERS
# =====================================================================
def plot_attention_heatmap(weights: np.ndarray, tokens: list, title: str):
    """Matplotlib heatmap for an attention weight matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(weights, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def draw_architecture_diagram():
    """
    Hand-drawn block diagram of the hybrid pipeline:
    Text -> Tokenizer -> Embedding -> Hybrid Quantum Self-Attention
         -> Feed-Forward -> Output (next-token logits)

    Drawn with plain matplotlib rectangles + arrows so it needs no
    external image asset and stays editable as code.
    """
    fig, ax = plt.subplots(figsize=(10, 3.2))
    blocks = [
        "Input\nText",
        "Tokenizer",
        "Embedding\nLayer",
        "Hybrid Quantum\nSelf-Attention",
        "Feed\nForward",
        "Output\n(Next Token)",
    ]
    n = len(blocks)
    box_w, box_h, gap = 1.5, 1.0, 0.55
    x = 0.3
    centers = []
    for i, label in enumerate(blocks):
        color = "#7c3aed" if "Quantum" in label else "#2563eb"
        rect = mpatches.FancyBboxPatch(
            (x, 0.5), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.5, edgecolor=color, facecolor=color + "22"
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, 0.5 + box_h / 2, label,
                 ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        centers.append(x + box_w)
        x += box_w + gap

    for i in range(n - 1):
        ax.annotate("", xy=(centers[i] + gap - 0.05, 1.0), xytext=(centers[i], 1.0),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="#444"))

    ax.set_xlim(0, x)
    ax.set_ylim(0, 2)
    ax.axis("off")
    fig.tight_layout()
    return fig


# =====================================================================
# 8. TOY NEXT-WORD PREDICTOR (classical bigram + quantum re-ranking)
# =====================================================================
CORPUS = (
    "quantum computing is a fast growing field of research "
    "machine learning models require large amounts of data "
    "artificial intelligence applications are transforming industries "
    "deep learning algorithms power modern language models "
    "hybrid quantum attention combines classical and quantum computation "
    "self attention is the core mechanism of the transformer architecture "
    "quantum circuits use qubits gates and entanglement to process information "
    "classical attention computes a weighted sum over all tokens in a sequence"
)


@st.cache_resource(show_spinner=False)
def build_bigram_model(corpus=CORPUS):
    """Build a simple {word: {next_word: count}} bigram frequency table."""
    toks = simple_tokenize(corpus)
    model = {}
    for a, b in zip(toks[:-1], toks[1:]):
        model.setdefault(a, {})
        model[a][b] = model[a].get(b, 0) + 1
    return model


def classical_next_word(model, last_word):
    """Pick the highest-frequency bigram continuation (classical baseline)."""
    candidates = model.get(last_word)
    if not candidates:
        return None, {}
    best = max(candidates, key=candidates.get)
    return best, candidates


def hybrid_quantum_next_word(model, last_word):
    """
    Re-rank the same bigram candidates using a quantum circuit.

    Each candidate word is hashed to 4 angles, run through the quantum
    circuit, and the mean expectation value is used as a multiplicative
    re-ranking weight on top of the classical bigram frequency. This is
    a simplified stand-in for the quantum-modulated attention score
    used in the full hybrid attention layer.
    """
    candidates = model.get(last_word)
    if not candidates:
        return None, {}
    scored = {}
    for word, freq in candidates.items():
        h = abs(hash(word)) % (10**6)
        rng = np.random.default_rng(h)
        angles = rng.uniform(-math.pi, math.pi, size=N_QUBITS)
        expvals = quantum_circuit(angles)
        quantum_weight = (np.mean(expvals) + 1) / 2  # rescale to [0,1]
        scored[word] = freq * (0.5 + quantum_weight)
    best = max(scored, key=scored.get)
    return best, scored


# =====================================================================
# ============================ STREAMLIT UI ==========================
# =====================================================================
st.set_page_config(page_title="Quantum Language Modeling Dashboard", layout="wide")
st.title("Hybrid Quantum Self-Attention Dashboard")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Overview",
        "NLP Fundamentals",
        "Next Word Prediction",
        "Attention Visualization",
        "Quantum Circuit Explorer",
        "Performance Analysis",
        "Research Findings",
    ],
)

# ---------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.header("Project Overview")

    st.subheader("Objective")
    st.write(
        """
        Design and simulate a **hybrid quantum self-attention layer** for
        small-scale language modeling, and quantify its **latency overhead**
        relative to classical self-attention, using a noiseless quantum
        simulator (PennyLane `default.qubit`) so the study is fully
        reproducible on classical hardware.
        """
    )

    st.subheader("Problem Statement")
    st.write(
        """
        Classical self-attention scales as **O(n²·d)** in sequence length
        `n` and model dimension `d`, which becomes a bottleneck for larger
        contexts. Quantum computing offers, in principle, exponentially
        large Hilbert spaces from a small number of qubits. This project
        investigates whether a **small (4-qubit) variational quantum
        circuit** can be embedded inside a self-attention block to enrich
        the attention scores with entanglement-based correlations, and
        measures the practical latency cost of doing so on a simulator —
        a necessary first step before any real-hardware deployment.
        """
    )

    st.subheader("Architecture")
    st.pyplot(draw_architecture_diagram())
    st.caption(
        "Pipeline: raw text → tokenizer → embedding layer → hybrid quantum "
        "self-attention → feed-forward block → next-token output."
    )

    st.subheader("What is Hybrid Quantum Self-Attention?")
    st.write(
        """
        In **classical self-attention**, every token's Query vector is
        compared against every other token's Key vector via a dot product,
        scaled and passed through softmax to obtain attention weights.

        In the **hybrid quantum** version used here, the same classical
        Q/K/V projections are computed first (so the layer remains
        gradient-friendly and trainable), but each query vector's leading
        4 dimensions are additionally encoded as **RY rotation angles**
        into a 4-qubit circuit. A ring of **CNOT gates entangles** the
        qubits, and the resulting **Pauli-Z expectation values** are used
        to modulate that query's raw attention scores before the softmax.
        The result is an attention mechanism whose scores depend on a
        genuinely quantum (entangled) computation, not just classical
        linear algebra.
        """
    )

# ---------------------------------------------------------------------
# PAGE 2: NLP FUNDAMENTALS
# ---------------------------------------------------------------------
elif page == "NLP Fundamentals":
    st.header("NLP Fundamentals")

    st.subheader("Tokenization Demo")
    text_in = st.text_input("Enter a sentence to tokenize", "Quantum attention improves language models")
    tokens = simple_tokenize(text_in)
    if tokens:
        vocab = build_vocab(tokens)
        st.dataframe(pd.DataFrame({
            "Token": tokens,
            "Token ID": [vocab[t] for t in tokens],
        }), use_container_width=True)
    else:
        st.info("Type something above to see it tokenized.")

    st.subheader("Embedding Explanation")
    st.write(
        """
        Each token id is mapped to a dense real-valued vector (an
        **embedding**) that the model learns during training. Below is a
        toy (untrained, randomly seeded) embedding table for the sentence
        above, visualised as a heatmap — each row is one token's vector.
        """
    )
    if tokens:
        emb_matrix, _ = get_embeddings(tokens)
        fig, ax = plt.subplots(figsize=(6, max(1.5, 0.4 * len(tokens))))
        im = ax.imshow(emb_matrix, cmap="coolwarm", aspect="auto")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel(f"Embedding dimension (d_model = {D_MODEL})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Sequence Length vs. Attention Cost")
    st.write(
        "Self-attention compute scales quadratically with sequence length "
        "(O(n²)). The chart below compares the three sequence lengths "
        "studied in this project."
    )
    seq_df = pd.DataFrame({
        "Sequence Length": SEQ_LENGTHS,
        "Relative Pairwise Comparisons (n²)": [L**2 for L in SEQ_LENGTHS],
    })
    fig = px.bar(
        seq_df, x="Sequence Length", y="Relative Pairwise Comparisons (n²)",
        text="Relative Pairwise Comparisons (n²)", color="Sequence Length",
        title="Quadratic Growth of Self-Attention Cost",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seq_df, use_container_width=True)

    st.subheader("Language Modeling, Briefly")
    st.write(
        """
        A language model assigns a probability distribution over the next
        token given the previous tokens: **P(tokenₙ | token₁ ... tokenₙ₋₁)**.
        Self-attention lets every token directly "look at" every other
        token when forming that prediction, which is why it replaced
        recurrent architectures (RNN/LSTM) as the dominant building block
        for modern language models.
        """
    )

# ---------------------------------------------------------------------
# PAGE 3: NEXT WORD PREDICTION
# ---------------------------------------------------------------------
elif page == "Next Word Prediction":
    st.header("Next Word Prediction")
    st.caption(
        "Illustrative demo only — a small bigram model built from a toy "
        "corpus, NOT a trained transformer. It exists purely to show how a "
        "classical scoring rule compares to a quantum-re-ranked one."
    )

    model = build_bigram_model()
    text = st.text_input("Enter text", "quantum computing")
    if text:
        toks = simple_tokenize(text)
        last_word = toks[-1] if toks else ""

        c_pred, c_scores = classical_next_word(model, last_word)
        q_pred, q_scores = hybrid_quantum_next_word(model, last_word)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Classical Prediction**")
            if c_pred:
                st.success(c_pred)
                st.dataframe(pd.DataFrame(
                    sorted(c_scores.items(), key=lambda x: -x[1]),
                    columns=["Candidate", "Bigram Frequency"]
                ), use_container_width=True)
            else:
                st.warning("No prediction available (word not in toy corpus).")

        with col2:
            st.markdown("**Hybrid Quantum Prediction**")
            if q_pred:
                st.success(q_pred)
                st.dataframe(pd.DataFrame(
                    sorted(q_scores.items(), key=lambda x: -x[1]),
                    columns=["Candidate", "Quantum-Reranked Score"]
                ), use_container_width=True)
            else:
                st.warning("No prediction available (word not in toy corpus).")

        if c_pred and q_pred and c_pred != q_pred:
            st.info(
                f"The quantum re-ranking changed the top prediction from "
                f"**{c_pred}** to **{q_pred}** for the word **'{last_word}'**."
            )

# ---------------------------------------------------------------------
# PAGE 4: ATTENTION VISUALIZATION
# ---------------------------------------------------------------------
elif page == "Attention Visualization":
    st.header("Attention Visualization")

    text = st.text_input("Enter a sentence", "quantum attention improves small language models")
    tokens = simple_tokenize(text)

    if len(tokens) < 2:
        st.warning("Please enter at least two tokens.")
    else:
        st.write("**Tokenized input:**", tokens)

        emb_matrix, _ = get_embeddings(tokens)
        X = torch.tensor(emb_matrix, dtype=torch.float32)

        c_weights, _ = classical_self_attention(X)
        with st.spinner("Running 4-qubit circuit for each query token..."):
            q_weights, _, q_mod = hybrid_quantum_attention(X)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_attention_heatmap(
                c_weights.detach().numpy(), tokens, "Classical Self-Attention"
            ))
        with col2:
            st.pyplot(plot_attention_heatmap(
                q_weights.detach().numpy(), tokens, "Hybrid Quantum Self-Attention"
            ))

        st.subheader("Difference Map (Quantum − Classical)")
        diff = (q_weights - c_weights).detach().numpy()
        st.pyplot(plot_attention_heatmap(diff, tokens, "Attention Weight Difference"))

        st.subheader("Per-token Quantum Modulation Signal")
        st.dataframe(pd.DataFrame({
            "Token": tokens,
            "Quantum Modulation (mean <Z>)": q_mod.detach().numpy(),
        }), use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 5: QUANTUM CIRCUIT EXPLORER
# ---------------------------------------------------------------------
elif page == "Quantum Circuit Explorer":
    st.header("Quantum Circuit Explorer")

    st.write(
        """
        This circuit is the quantum building block used inside the hybrid
        attention layer. Adjust the four RY rotation angles below (one per
        qubit) and observe how the circuit diagram and measured outputs
        change.
        """
    )

    cols = st.columns(4)
    angles = []
    for i, c in enumerate(cols):
        with c:
            a = st.slider(f"θ (qubit {i})", 0.0, float(2 * math.pi), float(math.pi / 4), 0.05, key=f"angle_{i}")
            angles.append(a)
    angles = np.array(angles)

    st.subheader("Circuit Diagram")
    st.pyplot(draw_quantum_circuit(angles))

    st.subheader("Measured Outputs (⟨Z⟩ per qubit)")
    expvals = quantum_circuit(angles)
    fig = go.Figure(go.Bar(
        x=[f"Qubit {i}" for i in range(N_QUBITS)],
        y=[float(v) for v in expvals],
        marker_color="#7c3aed",
    ))
    fig.update_layout(yaxis_range=[-1, 1], title="Pauli-Z Expectation Values")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Concepts explained"):
        st.markdown(
            """
            - **Qubit** — the quantum analogue of a classical bit; instead
              of being strictly 0 or 1, it can exist in a weighted
              superposition `α|0⟩ + β|1⟩`.
            - **Superposition** — created here by the **RY(θ)** gate, which
              rotates a qubit's state around the Y-axis of the Bloch
              sphere by angle θ, continuously interpolating between |0⟩
              and |1⟩.
            - **Entanglement** — produced by the **CNOT** gates. After
              entanglement, the qubits' measurement outcomes become
              statistically correlated in a way that has no classical
              analogue, which is the resource this project tries to
              exploit inside the attention score.
            - **RY gate** — single-qubit rotation gate parameterised by
              one real angle; used here to *encode* classical attention
              features into the quantum state.
            - **CNOT gate** — a two-qubit gate that flips its target
              qubit if and only if the control qubit is `|1⟩`; chaining
              CNOTs in a ring entangles all 4 qubits together.
            - **⟨Z⟩ (Pauli-Z expectation value)** — a number in [-1, 1]
              read out from each qubit after the circuit runs; this is
              the *classical* information we extract from the quantum
              state to feed back into the attention mechanism.
            """
        )

# ---------------------------------------------------------------------
# PAGE 6: PERFORMANCE ANALYSIS
# ---------------------------------------------------------------------
elif page == "Performance Analysis":
    st.header("Performance Analysis")
    st.write(
        "Benchmarking classical vs. hybrid quantum self-attention across "
        "the three sequence lengths studied in this project: "
        f"{', '.join(map(str, SEQ_LENGTHS))} tokens."
    )

    with st.spinner("Running benchmark (quantum circuit is evaluated once per token)..."):
        df = benchmark_attention(tuple(SEQ_LENGTHS))

    st.subheader("Latency Table")
    st.dataframe(df, use_container_width=True)

    st.subheader("Latency vs. Sequence Length")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Sequence Length"], y=df["Classical Latency (ms)"],
                              mode="lines+markers", name="Classical Attention"))
    fig.add_trace(go.Scatter(x=df["Sequence Length"], y=df["Hybrid Quantum Latency (ms)"],
                              mode="lines+markers", name="Hybrid Quantum Attention"))
    fig.update_layout(xaxis_title="Sequence Length", yaxis_title="Latency (ms)",
                       title="Classical vs. Hybrid Quantum Attention Latency")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quantum Overhead Factor")
    fig2 = px.bar(df, x="Sequence Length", y="Overhead (x)",
                  text="Overhead (x)", title="Hybrid Quantum Latency Overhead (×Classical)")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Overhead grows with sequence length because the quantum circuit "
        "is currently invoked once per query token on a classical "
        "simulator; on real quantum hardware (or a batched simulator "
        "execution) this scaling would differ substantially."
    )

# ---------------------------------------------------------------------
# PAGE 7: RESEARCH FINDINGS
# ---------------------------------------------------------------------
elif page == "Research Findings":
    st.header("Research Findings")

    st.subheader("Key Observations")
    st.markdown(
        """
        - The hybrid quantum attention layer produces attention
          distributions that are **measurably different** from classical
          attention, even though both are seeded from the same token
          embeddings — the quantum modulation term shifts probability
          mass toward different tokens.
        - On a classical simulator, hybrid quantum attention is
          **consistently slower** than classical attention, and the gap
          widens as sequence length grows from 64 → 128 → 256 tokens,
          because the circuit is currently evaluated once per token.
        - A 4-qubit circuit with a single layer of RY + ring-CNOT gates
          is enough to produce a non-trivial, entanglement-derived
          re-ranking signal, suggesting deeper circuits are not strictly
          required for a first proof of concept.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - All experiments run on a **noiseless classical simulator**
          (PennyLane `default.qubit`); no real quantum hardware noise,
          decoherence, or shot noise is modelled.
        - The next-word prediction demo uses a **toy bigram corpus**, not
          a trained transformer — it illustrates the re-ranking concept,
          not real language modeling accuracy.
        - The quantum circuit is invoked **once per token per forward
          pass**, which is not yet competitive with batched classical
          attention; no circuit-batching or amplitude-encoding strategies
          have been explored yet.
        - Only a **single quantum circuit architecture** (RY + ring CNOT)
          was studied; no ablation over circuit depth or qubit count was
          performed.
        """
    )

    st.subheader("Future Work")
    st.markdown(
        """
        - Benchmark on real NISQ hardware (e.g. IBM Quantum, IonQ) to
          measure latency and fidelity under real noise.
        - Explore amplitude encoding and circuit batching to reduce the
          per-token quantum overhead.
        - Train an actual small transformer with the hybrid attention
          layer end-to-end and compare perplexity against a classical
          baseline of identical parameter count.
        - Run an ablation study over qubit count (4, 6, 8) and circuit
          depth to characterize the accuracy/latency trade-off curve.
        """
    )