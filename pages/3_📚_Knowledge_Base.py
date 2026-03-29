"""
Knowledge Base Browser — Browse indexed documents and search the RAG corpus.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import RCA_PLAYBOOKS_DIR, CORPUS_DIR

st.set_page_config(page_title="Knowledge Base", page_icon="📚", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .page-header h2 { color: white; margin: 0; }
    .page-header p { color: #e0e0e0; margin: 0.3rem 0 0 0; }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h2>📚 Knowledge Base Browser</h2>
    <p>Browse and search the RAG corpus used by the Investigator Agent</p>
</div>
""", unsafe_allow_html=True)

# ── KB Status ──
try:
    from src.rag.knowledge_base import KnowledgeBase, build_knowledge_base
    kb = KnowledgeBase()
    kb_count = kb.count
except Exception as e:
    kb_count = 0
    st.warning(f"Knowledge base not initialized: {e}")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Chunks", kb_count)
with c2:
    playbook_count = len(list(RCA_PLAYBOOKS_DIR.glob("*.md"))) if RCA_PLAYBOOKS_DIR.exists() else 0
    st.metric("Playbooks", playbook_count)
with c3:
    try:
        sources = kb.get_all_sources()
        st.metric("Source Documents", len(sources))
    except Exception:
        st.metric("Source Documents", 0)

# ── Build/Rebuild ──
st.markdown("---")
b1, b2 = st.columns(2)
with b1:
    if st.button("🔨  Build Knowledge Base", type="primary", use_container_width=True):
        with st.spinner("Building knowledge base..."):
            kb = build_knowledge_base(force_rebuild=False)
            st.success(f"Ready — {kb.count} chunks indexed.")
            st.rerun()
with b2:
    if st.button("🔄  Rebuild from Scratch", use_container_width=True):
        with st.spinner("Rebuilding knowledge base..."):
            kb = build_knowledge_base(force_rebuild=True)
            st.success(f"Rebuilt — {kb.count} chunks indexed.")
            st.rerun()

# ── Search ──
st.markdown("---")
st.markdown("### Search Knowledge Base")

s1, s2 = st.columns([3, 1])
with s1:
    search_query = st.text_input("Search query", placeholder="e.g., zero billing root cause CDR failure")
with s2:
    search_k = st.slider("Results", 1, 20, 5)

if search_query:
    if kb_count > 0:
        results = kb.search(search_query, n_results=search_k)
        st.markdown(f"**{len(results)} results**")
        for i, r in enumerate(results, 1):
            pct = r["relevance_score"] * 100
            with st.expander(f"#{i} — {r['source']}  ({pct:.1f}% relevance)", expanded=(i <= 2)):
                st.markdown(f"**Source:** `{r['source']}` | **Score:** {r['relevance_score']:.4f}")
                st.write(r["text"])
                if r.get("metadata"):
                    st.json(r["metadata"])
    else:
        st.warning("Knowledge base is empty. Click **Build Knowledge Base** first.")

# ── Playbooks ──
st.markdown("---")
st.markdown("### RCA Playbooks")

if RCA_PLAYBOOKS_DIR.exists():
    playbooks = sorted(RCA_PLAYBOOKS_DIR.glob("*.md"))
    if playbooks:
        for pb in playbooks:
            with st.expander(f"📄 {pb.stem.replace('_', ' ').title()}"):
                st.markdown(pb.read_text(encoding="utf-8"))
    else:
        st.info("No playbooks found. Run system setup first.")
else:
    st.info("Playbooks directory not found.")

# ── Indexed Sources ──
st.markdown("---")
st.markdown("### Indexed Sources")
if kb_count > 0:
    try:
        sources = kb.get_all_sources()
        for src in sources:
            st.markdown(f"- 📄 `{src}`")
    except Exception as e:
        st.warning(f"Could not retrieve sources: {e}")
else:
    st.info("No documents indexed yet.")
