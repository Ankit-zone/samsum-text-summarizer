
import streamlit as st
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SamSum AI | Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "samsum_t5_final"

# Handle an extra nested folder after ZIP extraction
if not (MODEL_PATH / "config.json").exists():
    configs = list(MODEL_PATH.rglob("config.json"))

    if configs:
        MODEL_PATH = configs[0].parent

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 1.7rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #e8edff, #f4eaff);
    color: #172554;
    margin-bottom: 1.5rem;
}
.hero h1 {
    margin: 0;
    font-size: 2.2rem;
}
.hero p {
    margin-top: 0.6rem;
    font-size: 1.05rem;
}
.stButton button {
    border-radius: 10px;
    min-height: 2.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Model configuration not found in {model_path}. "
            "Check that the extracted model folder contains config.json."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        local_files_only=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📝 SamSum AI Summarizer</h1>
    <p>Turn lengthy conversations into clear, concise summaries
    using your fine-tuned Transformer model.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Settings")

    summary_length = st.select_slider(
        "Summary length",
        options=["Short", "Balanced", "Detailed"],
        value="Balanced"
    )

    num_beams = st.slider(
        "Generation quality",
        min_value=1,
        max_value=4,
        value=4,
        help="More beams can improve output but take longer."
    )

    st.divider()
    st.subheader("About")
    st.write(
        "This app uses a fine-tuned T5 encoder-decoder model "
        "trained for dialogue summarization on SAMSum."
    )

    st.caption("Model inference runs on the application server.")

# ---------------- INPUT ----------------
left, right = st.columns(2, gap="large")

example_text = (
    "Alex: Are we meeting for the project today?\n"
    "Priya: Yes, at 4 PM in the library.\n"
    "Alex: Should I bring the presentation?\n"
    "Priya: Yes, please bring the slides."
)

with left:
    st.subheader("💬 Conversation")
    st.caption("Paste the conversation you want to summarize.")

    if st.button("Load example conversation"):
        st.session_state["dialogue"] = example_text

    dialogue = st.text_area(
        "Conversation input",
        key="dialogue",
        height=330,
        placeholder="Alex: Hello!\nPriya: Hi, how are you?",
        label_visibility="collapsed"
    )

    word_count = len(dialogue.split())
    st.caption(f"📝 {word_count} words")

    generate = st.button(
        "✨ Generate Summary",
        type="primary",
        use_container_width=True
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["dialogue"] = ""
        st.session_state.pop("summary_result", None)
        st.rerun()

# ---------------- GENERATION ----------------
if generate:
    if not dialogue.strip():
        st.warning("Please enter a conversation first.")
    elif word_count < 3:
        st.warning("Please enter a longer conversation.")
    else:
        try:
            with st.spinner("Loading model and generating summary..."):
                tokenizer, model, device = load_model(MODEL_PATH)

                max_input_tokens = 512
                input_text = "summarize: " + dialogue.strip()

                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=max_input_tokens,
                    truncation=True
                ).to(device)

                lengths = {
                    "Short": (30, 60),
                    "Balanced": (50, 100),
                    "Detailed": (80, 140)
                }

                min_tokens, max_tokens = lengths[summary_length]

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        min_new_tokens=min_tokens,
                        num_beams=num_beams,
                        do_sample=False,
                        early_stopping=True
                    )

                summary = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                ).strip()

                st.session_state["summary_result"] = summary

        except Exception as e:
            st.error(f"Could not generate summary: {e}")

# ---------------- OUTPUT ----------------
with right:
    st.subheader("✨ Generated Summary")
    st.caption("Your AI-generated result will appear here.")

    summary = st.session_state.get("summary_result", "")

    if summary:
        st.text_area(
            "Summary output",
            value=summary,
            height=260,
            key="summary_display"
        )

        summary_words = len(summary.split())
        original_words = len(dialogue.split())

        col1, col2 = st.columns(2)
        col1.metric("Original words", original_words)
        col2.metric("Summary words", summary_words)

        if original_words:
            reduction = max(
                0,
                (1 - summary_words / original_words) * 100
            )
            st.metric("Word reduction", f"{reduction:.1f}%")

        st.download_button(
            "⬇️ Download summary (.txt)",
            data=summary,
            file_name="samsum_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info(
            "Your summary will appear here after you click "
            "'Generate Summary'."
        )

st.divider()
st.caption(
    "Built with Streamlit, PyTorch, and Hugging Face Transformers."
)
