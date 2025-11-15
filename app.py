import streamlit as st
from pipeline import Pipeline
from datetime import datetime

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Privacy-First Transaction Classifier",
    layout="wide",
    page_icon="🛡️"
)

# --------------------------------------------------------
# DARK MODE STYLE
# --------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #0e0e0e;
    color: #e6e6e6;
}

.main {
    background-color: #0e0e0e !important;
}

.big-title {
    font-size: 38px;
    font-weight: 900;
    color: #f5d98b;
    text-align: center;
    margin-bottom: -5px;
    text-shadow: 0 0 12px rgba(255,225,150,0.6);
}

.subtitle {
    font-size: 17px;
    text-align: center;
    color: #bbbbbb;
    margin-bottom: 25px;
}

.result-card {
    background: rgba(255,255,255,0.04);
    padding: 24px;
    border-radius: 14px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.65);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 24px;
}

.badge {
    padding: 5px 12px;
    border-radius: 7px;
    font-weight: 700;
    font-size: 14px;
    color: black;
    text-shadow: 0 0 4px rgba(0,0,0,0.3);
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 15px;
    color: #fce8a6;
}

.small-text {
    font-size: 14px;
    color: #aaaaaa;
}

textarea {
    background-color: #1a1a1a !important;
    color: #e6e6e6 !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.markdown('<p class="big-title">🛡️ Privacy-First Transaction Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Secure • Offline • ML + Rule Hybrid Engine</p>', unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------
# LOAD PIPELINE
# --------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return Pipeline()

pipeline = load_pipeline()

# --------------------------------------------------------
# CATEGORY COLORS — DARK MODE GLOW
# --------------------------------------------------------
CATEGORY_COLORS = {
    "Shopping": "#ffb3c1",
    "Fuel": "#ffda7b",
    "Dining": "#9dffb0",
    "Groceries": "#d1ff9f",
    "Travel": "#7fd3ff",
    "Subscriptions": "#d8b4ff",
    "Medical": "#ffb3b3",
    "Utilities": "#f7e299",
    "EMI/Loan": "#ff9ecd",
    "Wallet": "#afffef",
    "ATM": "#ffd3a1",
    "Fees": "#ff9e9e",
    "Transfers": "#b2d7ff",
    "Finance": "#a5fff5",
}

def badge(label):
    color = CATEGORY_COLORS.get(label, "#e6e6e6")
    return f'<span class="badge" style="background:{color};">{label}</span>'

# --------------------------------------------------------
# INPUT AREA
# --------------------------------------------------------
st.markdown("### <span style='color:#f5d98b;'>Enter Transactions</span>", unsafe_allow_html=True)
st.markdown('<p class="small-text">Each line is classified separately.</p>', unsafe_allow_html=True)

default_text = """UPI AMAZON PAY 499
UPI BIGBAZAAR 599
UPI ZOMATO 299
HP PETROL PUMP 1200
NETFLIX SUBSCRIPTION 499
UPI AMAZON PAY 499
UPI BIGBAZAAR 599
UPI ZOMATO 299
HP PETROL PUMP 1200
NETFLIX SUBSCRIPTION 499
SWIGGY ORDER 425
ZOMATO ORDER 310
AMAZON PRIME MEMBERSHIP 1499
DMART PURCHASE 850
RELIANCE FRESH GROCERY 560
ATM WITHDRAWAL 5000
SERVICE CHARGE 50
ELECTRICITY BILL TSSPDCL 1650
UPI TRANSFER TO SAVINGS 10000
PAID TO KFC 445
MCDONALDS ORDER 330
BPCL FUEL STATION 950
SPOTIFY SUBSCRIPTION 199
AJIO SHOPPING 899
FLIPKART ORDER 1299
WALLET TOPUP PAYTM 200
NEFT FUND TRANSFER 7500
CREDIT CARD EMI ICICI BANK 3500
HOTSTAR SUBSCRIPTION 499
"""

text = st.text_area("Transaction Text", height=150, value=default_text)

# --------------------------------------------------------
# CLASSIFY BUTTON
# --------------------------------------------------------
if st.button("🔍 Classify Transactions", use_container_width=True):

    if not pipeline:
        st.error("Pipeline failed to load.")
    else:
        st.write("")

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines:
            with st.spinner(f"Processing: {line}"):
                result = pipeline.classify(line)

            # --------------------------------------------------------
            # RESULT CARD
            # --------------------------------------------------------
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            st.markdown(f"<h3 style='color:#ffe9a6;'>🔹 Transaction: <code>{line}</code></h3>", unsafe_allow_html=True)

            st.markdown(badge(result['category']), unsafe_allow_html=True)

            st.markdown(f"""
            <div class='small-text'>
                <b style='color:#fce8a6;'>Confidence:</b> {result.get('confidence')}<br>
                <b style='color:#fce8a6;'>Source:</b> {result.get('source')}<br>
                <b style='color:#fce8a6;'>Cleaned:</b> {result.get('cleaned')}
            </div>
            """, unsafe_allow_html=True)

            if result.get("explanation"):
                st.markdown("<p class='section-title'>Rule Triggered</p>", unsafe_allow_html=True)
                st.json(result["explanation"])

            if result.get("top_features"):
                st.markdown("<p class='section-title'>Top Influencing Tokens</p>", unsafe_allow_html=True)
                for tok, weight in result["top_features"]:
                    st.markdown(f"- **{tok}** → `{weight:.4f}`")

            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")
