import re
from typing import Tuple, Dict

# ===== REGEX PATTERNS =====
PHONE_RE = re.compile(r"(?:(?:\+91|0)?\s?)(?:\d{10}|\d{5}[-\s]\d{5})")
UPI_RE = re.compile(r"\b[\w\.\-]{2,}@[a-z]{2,}\b")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
CARD_RE = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")
IFSC_RE = re.compile(r"[A-Za-z]{4}0[A-Z0-9]{6}")
AADHAR_RE = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
GST_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9][Z][A-Z0-9]\b")
OTP_RE = re.compile(r"\b\d{4,6}\b")

# Capitalized Names (improved to avoid over-masking)
NAME_RE = re.compile(r"\b([A-Z][a-z]{3,})\b")


# ===== UTILS =====
def mask_pattern(text: str, pattern: re.Pattern, mask_token: str) -> str:
    return pattern.sub(mask_token, text)


def normalize_merchants(t: str) -> str:
    """
    Normalize merchant spellings (very important for accuracy)
    """
    replacements = {
        "amazn": "amazon",
        "amzn": "amazon",
        "flip cart": "flipkart",
        "big bazar": "bigbazaar",
        "bb instant": "blinkit",
        "zoma to": "zomato",
        "zomat o": "zomato",
        "mynta": "myntra",
        "rpay": "razorpay",
        "jio mart": "jiomart",
        "ola cabs": "ola",
        "uber india": "uber",
        "foodpanda": "foodpanda",
        "tata 1 mg": "tata 1mg",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    return t


# ===== MAIN PREPROCESS FUNCTION =====
def preprocess(text: str) -> Tuple[str, Dict]:
    """
    Masks PII, normalizes merchant names,
    and returns (cleaned_text, mask_info_dict)
    """
    original = text
    masks = {}

    t = text

    # Mask PII
    t = mask_pattern(t, PHONE_RE, "phone_masked")
    t = mask_pattern(t, UPI_RE, "upi_masked")
    t = mask_pattern(t, EMAIL_RE, "email_masked")
    t = mask_pattern(t, CARD_RE, "card_masked")
    t = mask_pattern(t, IFSC_RE, "ifsc_masked")
    t = mask_pattern(t, AADHAR_RE, "aadhar_masked")
    t = mask_pattern(t, PAN_RE, "pan_masked")
    t = mask_pattern(t, GST_RE, "gst_masked")
    t = mask_pattern(t, OTP_RE, "otp_masked")

    # Mask capitalized names (but not merchants)
    def mask_name(m):
        name = m.group(0)
        common_words = ["Amazon", "Uber", "Blinkit", "Domino", "IRCTC", "Ola"]
        return name if name in common_words else "name_masked"

    t = NAME_RE.sub(mask_name, t)

    # Normalize text (lowercase)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Merchant normalization
    t = normalize_merchants(t)

    masks["original"] = original
    masks["masked_text"] = t

    return t, masks


if __name__ == "__main__":
    examples = [
        "Paid to Rahul via UPI rahul@okhdfc 250",
        "Amazon marketplace AMZN payment 499",
        "Uber India ride payment 320",
    ]
    for ex in examples:
        print(preprocess(ex))
