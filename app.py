import streamlit as st
import torch
import json
import re
import os
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

BASE_MODEL  = "unsloth/DeepSeek-R1-Distill-Llama-8B"
LORA_MODEL  = "Nguuma/Fine-tuned-DeepSeek-R1-Lab-Test-Assistant-LoRA"
HF_TOKEN    = os.environ.get("HUGGINGFACE_TOKEN")
SYSTEM_MSG  = (
    "You are a medical laboratory assistant. Analyze lab results, "
    "identify abnormalities based on reference ranges, and provide "
    "clear, brief explanations for healthcare professionals."
)

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Lab Test Assistant",
    page_icon  = "🧬",
    layout     = "wide",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; color: #e0e0e0; }
    .stTextInput > div > div > input { background: #1e2130; color: #fff; border: 1px solid #3a3f5c; border-radius: 8px; }
    .stButton > button { background: #2563eb; color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600; }
    .stButton > button:hover { background: #1d4ed8; }
    .result-card  { background: #1e2130; border-radius: 12px; padding: 20px; margin: 12px 0; }
    .severity-critical { border-left: 4px solid #ef4444; }
    .severity-high     { border-left: 4px solid #f97316; }
    .severity-normal   { border-left: 4px solid #22c55e; }
    .severity-low      { border-left: 4px solid #3b82f6; }
    .chat-user     { background: #1e3a5f; border-radius: 12px; padding: 12px 16px; margin: 8px 0; }
    .chat-assistant { background: #1e2130; border-radius: 12px; padding: 12px 16px; margin: 8px 0; border-left: 4px solid #2563eb; }
    h1 { color: #60a5fa !important; }
    .stTabs [data-baseweb="tab"] { color: #9ca3af; }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Medical Lab Test Assistant")
st.caption(f"Powered by `{LORA_MODEL}` · Fine-tuned DeepSeek-R1-Distill-Llama-8B")

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fine-tuned model from HuggingFace…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token = HF_TOKEN,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit = torch.cuda.is_available(),  # 4-bit only on GPU
        torch_dtype  = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map   = "auto" if torch.cuda.is_available() else None,
        token        = HF_TOKEN,
    )
    model = PeftModel.from_pretrained(model, LORA_MODEL, token=HF_TOKEN)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()
gpu_label = f"🟢 GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "🟡 CPU (slow)"
st.sidebar.success(f"Model loaded · {gpu_label}")

# ── Inference Helper ──────────────────────────────────────────────────────────
def run_inference(query: str) -> dict:
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
{SYSTEM_MSG}

### Question:
{query}

### Response:
<think>"""

    inputs  = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids          = inputs.input_ids,
            attention_mask     = inputs.attention_mask,
            max_new_tokens     = 300,
            do_sample          = True,
            temperature        = 0.7,
            repetition_penalty = 1.3,
            eos_token_id       = tokenizer.eos_token_id,
            use_cache          = True,
        )

    raw_output    = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = raw_output.split("### Response:")[-1].strip()

    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    reasoning   = think_match.group(1).strip() if think_match else ""
    answer      = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    sev_match = re.match(r"^(Critical|High|Normal|Low|Borderline)[:\s]*([^.]*\.?)", answer, re.IGNORECASE)
    severity  = sev_match.group(1).strip() if sev_match else "Unknown"
    summary   = sev_match.group(2).strip() if sev_match else answer[:120]

    recommendations = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", answer)
        if any(k in s.lower() for k in ["recommend", "suggest", "consider", "refer", "monitor", "prescribe", "urgent"])
    ]

    return {
        "severity":       severity,
        "summary":        summary,
        "full_response":  answer,
        "reasoning":      reasoning,
        "recommendations": recommendations,
    }


def severity_class(severity: str) -> str:
    s = severity.lower()
    if "critical" in s: return "severity-critical"
    if "high"     in s: return "severity-high"
    if "low"      in s: return "severity-low"
    return "severity-normal"


def render_result(query: str, result: dict, as_json: bool = False):
    css = severity_class(result["severity"])
    emoji = {"critical": "🔴", "high": "🟠", "low": "🔵", "normal": "🟢"}.get(result["severity"].lower(), "⚪")

    if as_json:
        st.json({
            "query":           query,
            "severity":        result["severity"],
            "summary":         result["summary"],
            "full_response":   result["full_response"],
            "reasoning":       result["reasoning"],
            "recommendations": result["recommendations"],
        })
        return

    st.markdown(f"""
    <div class="result-card {css}">
        <h4 style="margin:0 0 8px 0">{emoji} {result['severity']} — {result['summary']}</h4>
        <p style="color:#9ca3af; font-size:13px; margin:0 0 10px 0">🔬 Query: {query}</p>
        <p style="margin:0; line-height:1.7">{result['full_response']}</p>
    </div>
    """, unsafe_allow_html=True)

    if result["recommendations"]:
        with st.expander("📋 Recommendations"):
            for r in result["recommendations"]:
                st.markdown(f"- {r}")

    if result["reasoning"]:
        with st.expander("🧠 Chain of Thought"):
            st.markdown(f"*{result['reasoning']}*")


# ── PDF Helpers ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def parse_lab_lines(text: str) -> list:
    pattern = re.compile(
        r"([A-Za-z][A-Za-z\s\(\)/\-]+?)\s{1,10}([\d.]+)\s*(mg/dL|g/dL|mmol/L|U/L|mEq/L|IU/L|%|ng/mL|µg/dL|pg/mL|fl|fL|10\^[\d/]+)\s*[:\-]?\s*([\d.]+-[\d.]+)",
        re.IGNORECASE,
    )
    results = []
    for m in pattern.finditer(text):
        results.append({
            "test":  m.group(1).strip(),
            "value": m.group(2).strip(),
            "unit":  m.group(3).strip(),
            "range": m.group(4).strip(),
        })
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
show_json = st.sidebar.toggle("Show JSON output", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Base model:** `{BASE_MODEL}`")
st.sidebar.markdown(f"**LoRA adapters:** `{LORA_MODEL}`")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_pdf = st.tabs(["💬 Text Chat", "📄 PDF Upload"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Enter a lab result to analyze")
    st.markdown("**Format:** `Test Name. Result: VALUE UNIT. Ref Range: MIN-MAX UNIT.`")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Lab query",
                placeholder="e.g. Glucose (Fasting). Result: 155 mg/dL. Ref Range: 70-99 mg/dL.",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Analyze →", use_container_width=True)

    if submitted and user_input.strip():
        with st.spinner("Analyzing…"):
            result = run_inference(user_input.strip())
        st.session_state.chat_history.append({"query": user_input.strip(), "result": result})

    # Render history newest-first
    for item in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="chat-user">👤 {item["query"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-assistant">', unsafe_allow_html=True)
        render_result(item["query"], item["result"], as_json=show_json)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PDF UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pdf:
    st.subheader("Upload a PDF lab report")
    st.markdown("The app will auto-detect lab values and analyze each one.")

    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("Reading PDF…"):
            pdf_text = extract_text_from_pdf(uploaded_pdf)

        with st.expander("📃 Extracted raw text (click to view)"):
            st.text(pdf_text[:3000] + ("…" if len(pdf_text) > 3000 else ""))

        lab_lines = parse_lab_lines(pdf_text)

        if lab_lines:
            st.success(f"✅ Detected **{len(lab_lines)}** lab result(s). Click **Analyze All** to run.")
            st.dataframe(
                [{"Test": l["test"], "Result": f"{l['value']} {l['unit']}", "Ref Range": l["range"]} for l in lab_lines],
                use_container_width=True,
            )

            if st.button("🔬 Analyze All Results"):
                for lab in lab_lines:
                    query = f"{lab['test']}. Result: {lab['value']} {lab['unit']}. Ref Range: {lab['range']}."
                    with st.spinner(f"Analyzing {lab['test']}…"):
                        result = run_inference(query)
                    render_result(query, result, as_json=show_json)
                    st.divider()

        else:
            st.warning("⚠️ Could not auto-detect lab values from this PDF (scanned image or unusual format).")
            st.markdown("**Paste the relevant text below and type your questions in the Chat tab:**")
            manual_text = st.text_area("Paste lab report text here", value=pdf_text, height=250)
            if st.button("Analyze pasted text"):
                with st.spinner("Analyzing…"):
                    result = run_inference(manual_text[:800])
                render_result("Pasted PDF content", result, as_json=show_json)
