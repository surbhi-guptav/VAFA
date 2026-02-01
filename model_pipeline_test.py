from assisstants.processor.text_processor import TextProcessor
from assisstants.Classifier.text_classifier import TextClassifier
from assisstants.extractor.fields_extractor import ExtractFields
import re

text = "My name is dhani and i want to deposit 4000 rs and My account number is 293849348094."

p = TextProcessor()
c = TextClassifier()
e = ExtractFields() 

processed = p.process_text(text)
try:
    lbl, conf = c.classify(processed, return_prob=True)
except TypeError:
    lbl = c.classify(processed)
    conf = None

print("RAW INPUT:", text)
print("PROCESSED:", processed)
print("CLASSIFIER:", lbl, conf)
print("--- Extractor attempts per label ---")
for label in ["Name", "Phone Number", "Amount", "Account Number"]:
    try:
        res = e.extract(label, processed)
    except Exception as ex:
        res = f"ERROR: {ex}"
    print(f"{label}: {res}")

# Heuristic extraction (regex-based)
m_amount = re.search(r"(?:rs\.?|inr|rupees)\s*([0-9,]+(?:\.\d{1,2})?)", processed, re.I)
if not m_amount:
    m_amount = re.search(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b", processed)
print("HEURISTIC_AMOUNT:", m_amount.group(1) if m_amount else None)

m_acc = re.search(r"\b\d{6,18}\b", processed)
print("HEURISTIC_ACCOUNT:", m_acc.group(0) if m_acc else None)

name_pat = re.search(r"(?:my name is|i am|this is|myself)\s+([a-zA-Z']+(?:\s+[a-zA-Z']+)*)", text, re.I)
print("HEURISTIC_NAME:", name_pat.group(1) if name_pat else None)
