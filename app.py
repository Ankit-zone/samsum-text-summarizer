
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="SamSum AI | Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_ID = "iankityadav03/samsum-t5-summarizer"
MAX_INPUT_TOKENS = 512

# =====================================================
# CUSTOM CSS
# =====================================================
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
.stButton button, .stDownloadButton button {
    border-radius: 10px;
    min-height: 2.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# =====================================================
# SUMMARIZATION HELPERS
# =====================================================
def get_input_limit(tokenizer, model):
    """Determine a conservative input-token limit."""
    limits = [MAX_INPUT_TOKENS]

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        limits.append(tokenizer_limit)

    model_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        limits.append(model_limit)

    return min(limits) - 8


def split_into_chunks(text, tokenizer, chunk_size):
    """Split long conversations into manageable token chunks."""
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if not token_ids:
        return []

    chunks = []

    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start:start + chunk_size]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def summarize_once(
    text,
    tokenizer,
    model,
    device,
    input_limit,
    max_new_tokens,
    num_beams
):
    """Summarize one input that fits the model's input limit."""
    prompt = "summarize: " + text.strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=input_limit,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()


def generate_summary(
    text,
    tokenizer,
    model,
    device,
    max_new_tokens=80,
    num_beams=4
):
    """
    Summarize short and long conversations.
    Long conversations are summarized in chunks, then combined.
    """
    text = text.strip()

    if not text:
        raise ValueError("Please enter a conversation.")

    input_limit = get_input_limit(tokenizer, model)
    chunk_size = max(32, input_limit - 24)

    # Tokenize the complete conversation without truncating it.
    chunks = split_into_chunks(text, tokenizer, chunk_size)

    if not chunks:
        raise ValueError(
            "No readable text was found in the conversation."
        )

    # Direct summarization for a conversation that fits.
    if len(chunks) == 1:
        return summarize_once(
            chunks[0],
            tokenizer,
            model,
            device,
            input_limit,
            max_new_tokens,
            num_beams
        )

    # Summarize each part of a long conversation.
    partial_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        st.write(f"Processing conversation part {index} of {len(chunks)}...")

        partial = summarize_once(
            chunk,
            tokenizer,
            model,
            device,
            input_limit,
            max_new_tokens,
            num_beams
        )

        if partial:
            partial_summaries.append(partial)

    if not partial_summaries:
        return "The model could not generate a summary. Try different input."

    combined = " ".join(partial_summaries).strip()

    # If the combined summaries are too long, summarize them again.
    # Limit recursion to prevent repeated summarization loops.
    for _ in range(2):
        combined_ids = tokenizer.encode(
            combined,
            add_special_tokens=False
        )

        if len(combined_ids) <= chunk_size:
            break

        combined_chunks = split_into_chunks(
            combined,
            tokenizer,
            chunk_size
        )

        reduced_summaries = []

        for chunk in combined_chunks:
            result = summarize_once(
                chunk,
                tokenizer,
                model,
                device,
                input_limit,
                max_new_tokens,
                num_beams
            )

            if result:
                reduced_summaries.append(result)

        if not reduced_summaries:
            break

        combined = " ".join(reduced_summaries).strip()

    return combined or "The model returned an empty summary."


# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>📝 SamSum AI Summarizer</h1>
    <p>
        Transform conversations into clear, concise summaries
        using your fine-tuned Transformer model.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR SETTINGS
# =====================================================
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
        help="Higher beam values may improve generation but take longer."
    )

    st.divider()
    st.subheader("About this project")
    st.write(
        "A dialogue summarization application powered by a "
        "fine-tuned T5 model trained on the SAMSum dataset."
    )
    st.caption("Model hosted on Hugging Face Hub.")

# =====================================================
# EXAMPLE CONVERSATION
# =====================================================
example_text = (
    "Neha: It's Priya's birthday this Saturday. "
    "We should plan a surprise party for her.\n"
    "Riya: Absolutely! Let's book a table at our favorite "
    "restaurant for 7 PM.\n"
    "Karan: I'll order a chocolate cake and arrange some balloons.\n"
    "Neha: I'll invite our college friends, but please don't tell Priya.\n"
    "Riya: I'll collect ₹500 from everyone to cover the food "
    "and decorations.\n"
    "Karan: I'll pick up the cake at 5 PM and bring it to the restaurant.\n"
    "Neha: Let's all arrive by 6:30 PM so we can surprise her."
)

# =====================================================
# INPUT AND OUTPUT LAYOUT
# =====================================================
left, right = st.columns(2, gap="large")

with left:
    st.subheader("💬 Conversation")
    st.caption("Paste a conversation of any practical length.")

    # Keep state-changing buttons before the text-area widget.
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        if st.button("Load example", use_container_width=True):
            st.session_state["dialogue"] = example_text
            st.session_state.pop("summary_result", None)
            st.rerun()

    with button_col2:
        if st.button("Clear all", use_container_width=True):
            st.session_state["dialogue"] = ""
            st.session_state.pop("summary_result", None)
            st.rerun()

    if "dialogue" not in st.session_state:
        st.session_state["dialogue"] = ""

    dialogue = st.text_area(
        "Conversation input",
        key="dialogue",
        height=330,
        placeholder=(
            "Paste your conversation here...\n\n"
            "Alex: Hello!\n"
            "Priya: Hi, how are you?"
        ),
        label_visibility="collapsed"
    )

    word_count = len(dialogue.split())
    st.caption(f"📝 {word_count} words")

    generate = st.button(
        "✨ Generate Summary",
        type="primary",
        use_container_width=True
    )

# =====================================================
# GENERATE SUMMARY
# =====================================================
if generate:
    if not dialogue.strip():
        st.warning("Please enter a conversation first.")

    elif word_count < 3:
        st.warning("Please enter a longer conversation.")

    else:
        try:
            with st.spinner(
                "Loading model and summarizing conversation..."
            ):
                tokenizer, model, device = load_model()

                lengths = {
                    "Short": 40,
                    "Balanced": 80,
                    "Detailed": 120
                }

                max_new_tokens = lengths[summary_length]

                summary = generate_summary(
                    dialogue,
                    tokenizer,
                    model,
                    device,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams
                )

            if summary:
                st.session_state["summary_result"] = summary
                st.session_state["summary_source"] = dialogue
            else:
                st.warning(
                    "The model returned an empty summary. "
                    "Try another conversation."
                )

        except Exception as e:
            st.error("Could not generate the summary.")
            st.exception(e)

# =====================================================
# DISPLAY SUMMARY
# =====================================================
with right:
    st.subheader("✨ Generated Summary")
    st.caption("Your generated summary appears here.")

    summary = st.session_state.get("summary_result", "")

    if summary:
        st.text_area(
            "Summary output",
            value=summary,
            height=260,
            key="summary_display"
        )

        original_text = st.session_state.get(
            "summary_source", dialogue
        )

        original_words = len(original_text.split())
        summary_words = len(summary.split())

        col1, col2 = st.columns(2)
        col1.metric("Original words", original_words)
        col2.metric("Summary words", summary_words)

        if original_words:
            reduction = (
                1 - summary_words / original_words
            ) * 100

            st.metric(
                "Word reduction",
                f"{reduction:.1f}%"
            )

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

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    "Built with Streamlit, PyTorch, Hugging Face Transformers, and T5."
)
