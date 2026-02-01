from assisstants.processor.text_processor import TextProcessor
from assisstants.Classifier.text_classifier import TextClassifier
from assisstants.extractor.fields_extractor import ExtractFields
import re

inputs = [
    "My name is Alice",
    "Call me at 9876543210",
    "Transfer 5000 rupees",
    "Account number is 123456789012",
    "I want to deposit 10,000 INR",
    "Phone: +919876543210",
    "Name: Rahul",
    "Send 2 lakh rupees",
    "My phone is 98765 43210",
    "Account 000123456789",
    "What's your name?",
    "where are you going",
    "I am Sufi Gupta",
    "Please credit 1500 rs",
    "Debit Rs. 2500",
    "Account no 987654321098",
    "Pay 5k",
    "Call +1 415 555 2671",
    "My number is 7000012345",
    "This is John and my account is 445566778899",
]

p = TextProcessor()
c = TextClassifier()
e = ExtractFields()

# Canonicalize label function (mirrors app.py logic)
def canonicalize_label(lbl: str | None):
    if not lbl:
        return None
    l = lbl.lower().strip().replace('_', ' ')
    if 'phone' in l or 'mobile' in l or 'contact' in l:
        return 'Phone Number'
    if 'amount' in l or 'money' in l or 'rupee' in l or 'rs' in l or 'price' in l:
        return 'Amount'
    if 'account' in l or 'acct' in l:
        return 'Account Number'
    if 'name' in l or 'person' in l:
        return 'Name'
    return None

# Question heuristic (mirror of app rule)
import logging

def is_question_like(text):
    try:
        tstrip = text.strip() if isinstance(text, str) else ""
        if re.search(r"^(where|what|who|why|how|when)\b", tstrip, re.I) or tstrip.endswith('?') or re.search(r"\b(where are you|how are you)\b", tstrip, re.I):
            return True
    except Exception:
        pass
    return False

print(f"{'INPUT':<60} | {'PRED_LABEL':<15} | {'CONF':<6} | {'EXTRACTED'}")
print('-'*110)
for txt in inputs:
    processed = p.process_text(txt)
    try:
        raw_label, conf = c.classify(processed, return_prob=True)
    except TypeError:
        raw_label = c.classify(processed)
        conf = None
    label = canonicalize_label(str(raw_label) if raw_label is not None else None)
    # override name if question-like
    if label == 'Name' and is_question_like(txt):
        label = None
    entity = None
    if label:
        try:
            extractor_result = e.extract(label, processed)
            if extractor_result:
                if isinstance(extractor_result, str):
                    entity = extractor_result.strip()
                elif isinstance(extractor_result, (list, tuple)):
                    for item in extractor_result:
                        if isinstance(item, str) and item.strip():
                            entity = item.strip(); break
                        if isinstance(item, dict):
                            for k in ('entity','value','text'):
                                if k in item and item[k]:
                                    entity = str(item[k]).strip(); break
                            if entity: break
                elif isinstance(extractor_result, dict):
                    for k in ('entity','value','text', label.lower().replace(' ', '_')):
                        if k in extractor_result and extractor_result[k]:
                            entity = str(extractor_result[k]).strip(); break
        except Exception as ex:
            entity = f"ERR:{ex}"
    # fallback heuristics if no extractor result
    if label and not entity:
        if label == 'Phone Number':
            m = re.search(r"(\+?\d[\d\s-]{8,}\d)", processed)
            if m:
                entity = re.sub(r"[\s-]", "", m.group(1))
        if label == 'Amount' and not entity:
            m = re.search(r"(?:rs\.?|inr|usd|dollars|rupees)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)", processed, re.IGNORECASE)
            if m:
                entity = m.group(1).replace(',', '')
        if label == 'Account Number' and not entity:
            m = re.search(r"\b\d{6,18}\b", processed)
            if m:
                entity = m.group(0)
        if label == 'Name' and not entity:
            name_pattern = re.search(r"(?:my name is|i am|this is|myself)\s+([a-zA-Z']+(?:\s+[a-zA-Z']+)*)", txt, re.IGNORECASE)
            if name_pattern:
                entity = name_pattern.group(1).strip().title()

    print(f"{txt[:60]:<60} | {str(label):<15} | {str(round(conf,3)) if isinstance(conf,float) else '':<6} | {entity}")
