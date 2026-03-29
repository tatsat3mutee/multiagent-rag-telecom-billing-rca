"""
Knowledge Base Browser — Browse indexed documents and search the RAG corpus.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import RCA_PLAYBOOKS_DIR, CORPUS_DIR

st.set_page_config(page_title="Knowledge Base", page_icon="📚", layout="wide")
st.title("📚 Knowledge Base Browser")
st.markdown("Browse and search the RAG knowledge base used by the Investigator Agent.")

# ── Knowledge Base Status ──
try:
    from src.rag.knowledge_base import KnowledgeBase, build_knowledge_base
    kb = KnowledgeBase()
    kb_count = kb.count
except Exception as e:
    kb_count = 0
    st.warning(f"Knowledge base not initialized: {e}")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📄 Total Chunks", kb_count)
with col2:
    playbook_count = len(list(RCA_PLAYBOOKS_DIR.glob("*.md"))) if RCA_PLAYBOOKS_DIR.exists() else 0
    st.metric("📖 Playbooks", playbook_count)
with col3:
    try:
        sources = kb.get_all_sources()
        st.metric("📁 Source Documents", len(sources))
    except Exception:
        st.metric("📁 Source Documents", 0)

# ── Build/Rebuild KB ──
st.markdown("---")
build_col1, build_col2 = st.columns(2)
with build_col1:
    if st.button("🔨 Build Knowledge Base"):
        with st.spinner("Building knowledge base..."):
            kb = build_knowledge_base(force_rebuild=False)
            st.success(f"Knowledge base ready: {kb.count} document chunks indexed.")
            st.rerun()
with build_col2:
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Rebuilding knowledge base from scratch..."):
            kb = build_knowledge_base(force_rebuild=True)
            st.success(f"Knowledge base rebuilt: {kb.count} document chunks indexed.")
            st.rerun()

# ── Search ──
st.markdown("---")
st.markdown("### 🔍 Search Knowledge Base")

search_query = st.text_input(
    "Search query",
    placeholder="e.g., zero billing root cause CDR failure",
)

search_k = st.slider("Number of results", 1, 20, 5)

if search_query:
    if kb_count > 0:
        results = kb.search(search_query, n_results=search_k)

        st.markdown(f"**{len(results)} results found:**")
        for i, r in enumerate(results, 1):
            relevance_pct = r['relevance_score'] * 100
            with st.expander(
                f"#{i} — {r['source']} (Relevance: {relevance_pct:.1f}%)",
                expanded=(i <= 2),
            ):
                st.markdown(f"**Source:** {r['source']}")
                st.markdown(f"**Relevance Score:** {r['relevance_score']:.4f}")
                st.markdown("**Content:**")
                st.write(r["text"])
                if r.get("metadata"):
                    st.json(r["metadata"])
    else:
        st.warning("Knowledge base is empty. Click 'Build Knowledge Base' first.")

# ── Browse Playbooks ──
st.markdown("---")
st.markdown("### 📖 RCA Playbooks")

if RCA_PLAYBOOKS_DIR.exists():
    playbooks = sorted(RCA_PLAYBOOKS_DIR.glob("*.md"))
    if playbooks:
        for pb in playbooks:
            with st.expander(f"📄 {pb.stem.replace('_', ' ').title()}"):
                content = pb.read_text(encoding="utf-8")
                st.markdown(content)
    else:
        st.info("No playbooks found. They will be created during system setup.")
else:
    st.info("Playbooks directory not found.")

# ── Source Documents ──
st.markdown("---")
st.markdown("### 📁 Indexed Sources")
if kb_count > 0:
    try:
        sources = kb.get_all_sources()
        for src in sources:
            st.markdown(f"- 📄 {src}")
    except Exception as e:
        st.warning(f"Could not retrieve sources: {e}")
else:
    st.info("No documents indexed yet.")
