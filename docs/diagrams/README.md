# Defense Deck — Architecture & Results Diagrams

Mermaid sources (paste into any Mermaid-compatible renderer — draw.io,
Mermaid Live Editor, GitHub markdown, or the VS Code Mermaid Preview).
Rendered PNGs can be committed alongside each source for the deck.

---

## D1. System Architecture (high level)

```mermaid
flowchart LR
    subgraph DATA["Data Layer"]
        R[(IBM Telco CSV)]
        CDR[(Telecom Italia CDR — Phase 1.5)]
        KB[/8 RCA Playbooks<br/>+ 3 Public Specs/]
    end
    subgraph DETECT["Detection"]
        IF[IsolationForest<br/>F1 0.81 · AUROC 0.877]
        DB[DBSCAN]
    end
    subgraph RAG["Retrieval"]
        VS[(ChromaDB<br/>all-MiniLM-L6-v2)]
        BM[BM25 Lexical]
        GR[GraphRAG<br/>NetworkX]
        RRF[RRF + Cross-Encoder]
    end
    subgraph AGENTS["Multi-Agent RCA (LangGraph)"]
        INV[Investigator]
        RE[Reasoner]
        CR[Critic]
        RP[Reporter]
    end
    EV[Evaluation<br/>ROUGE · BERTScore · RAGAS · LLM-Judge<br/>Bootstrap CI · Wilcoxon]

    R --> IF --> INV
    CDR -.-> IF
    KB --> VS & BM & GR
    VS & BM --> RRF
    RRF --> INV
    GR -.-> INV
    INV --> RE --> CR
    CR -- revise --> INV
    CR -- accept --> RP --> EV
```

---

## D2. LangGraph StateGraph (agent DAG)

```mermaid
stateDiagram-v2
    [*] --> Investigator
    Investigator --> broaden_query: retrieval_count<2
    Investigator --> Reasoner: proceed
    broaden_query --> Reasoner
    Reasoner --> Critic
    Critic --> broaden_query: revise (once)
    Critic --> Reporter: accept
    Reporter --> [*]
```

---

## D3. Evaluation Framework

```mermaid
flowchart TB
    subgraph TRAD["Traditional"]
        ROUGE[ROUGE-L]
        BERT[BERTScore]
    end
    subgraph RAGAS["RAGAS-style"]
        FAITH[Faithfulness<br/>claims / total]
        REL[Answer Relevancy<br/>reverse Q sim]
    end
    subgraph JUDGE["LLM-as-Judge (GPT-4o)"]
        CORR[Correctness]
        GR2[Groundedness]
        ACT[Actionability]
        COMP[Completeness]
    end
    subgraph STATS["Statistical"]
        CI[Bootstrap CI]
        WLX[Wilcoxon]
        PB[Paired Bootstrap p]
    end
    RCA[Generated RCA] --> TRAD & RAGAS & JUDGE
    TRAD & RAGAS & JUDGE --> STATS --> REPORT[Significance Report]
```

---

## D4. GraphRAG Entity Schema (playbook extraction)

```mermaid
classDiagram
    class SYSTEM {
        name
        chunks[]
    }
    class COMPONENT
    class FAILURE_MODE
    class FIX
    class METRIC
    SYSTEM --|> Node
    COMPONENT --|> Node
    FAILURE_MODE --|> Node
    FIX --|> Node
    METRIC --|> Node
    SYSTEM --> COMPONENT : FEEDS_INTO
    COMPONENT --> COMPONENT : DEPENDS_ON
    FAILURE_MODE --> FAILURE_MODE : CAUSES
    FAILURE_MODE --> FIX : FIXES
    METRIC --> COMPONENT : MONITORS
    FAILURE_MODE --> FAILURE_MODE : TRIGGERS
```

---

## D5. Ablation Configs

```mermaid
flowchart LR
    A[Config A<br/>No RAG · No agents]
    B[Config B<br/>RAG + LLM]
    C[Config C<br/>Single-Agent + RAG]
    D[Config D<br/>Multi-Agent + RAG]
    E[Config E<br/>Multi-Agent + GraphRAG<br/>★ headline]
    A --> B --> C --> D --> E
```

---

## D6. Rendering notes

Use `scripts/render_diagrams.py` (if/when created) to batch-convert these
Mermaid blocks to PNG via `mmdc`. For the deck itself:

- Export each diagram at 1600×900 (16:9) PNG.
- Save to `docs/diagrams/D{n}_{slug}.png`.
- Inline into the LaTeX/PowerPoint defense deck.

Bar charts / significance plots: use `scripts/plot_results.py` (matplotlib,
publication-ready; 10pt labels, grayscale-safe).
