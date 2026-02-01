import sys
from assisstants.exception.exception import AssisstantException
from assisstants.logging.logger import logging

# lazy-load spacy model to avoid repeated disk loads
_SPACY_NLP = None

class ExtractFields:
    def extract(self, label, text, processed_text=None):
        """
        Extract entity for a given `label` from `text`.
        `text` should be the original raw transcript; `processed_text` is optional
        preprocessed text where numbers may already be normalized.
        Returns a string when found, otherwise an empty list (for backward compatibility).
        """
        try:
            logging.info("Field Extraction Started")
            import re
            from assisstants.utils.main_utils import convert_words_to_numbers

            raw = text or ""
            proc = processed_text or raw

            # Phone number: Indian format = exactly 10 digits (6-9 start), optional +91 or 0 prefix
            if label == "Phone Number":
                # Prefer matches explicitly near phone keywords
                m_ph = re.search(r"(?:phone|call|contact|mobile|number)[^\d]{0,25}(\+91[\s\-]?)?([6-9]\d{9})", raw, re.IGNORECASE)
                if m_ph:
                    digits = m_ph.group(2)
                    return digits
                # Fallback: look for +91 followed by 10-digit phone (very strict)
                m_international = re.search(r"\+91[\s\-]?([6-9]\d{9})", raw, re.IGNORECASE)
                if m_international:
                    return m_international.group(1)
                # NO generic fallback: require explicit keyword or +91 prefix to avoid capturing account digits
                return []

            # Account numbers: long digit sequences (11-18 digits, NOT 10) - prefer near 'account' keyword
            if label == "Account Number":
                # look for 'account' followed by digits in range 11-18
                m = re.search(r"(?:account|acct|a/c)[^\d]{0,25}(\d{11,18})", raw, re.IGNORECASE)
                if m:
                    return m.group(1)
                # fallback: pick any 11-18 digit token (avoid 10-digit phones)
                accs = re.findall(r"\b\d{11,18}\b", raw)
                if accs:
                    # sort by length descending, pick longest (most likely account)
                    accs.sort(key=lambda s: len(s), reverse=True)
                    return accs[0]
                return []

            # Amounts: try processed_text first (numbers normalized), else raw with word conversion
            if label == "Amount":
                # Prefer amounts appearing near currency words or transfer verbs in raw text
                m = re.search(r"(?:deposit|transfer|credit|debit|pay|send)[^\d\n\r]{0,30}(?:Rs\.?|INR|₹)?\s*([0-9,]+(?:\.\d{1,2})?)", raw, re.IGNORECASE)
                if m:
                    val = m.group(1).replace(',', '')
                    if len(re.sub(r"[^0-9]", "", val)) < 10:
                        return val
                # currency marker anywhere
                m2 = re.search(r"(?:Rs\.?|INR|₹)\s*([0-9,]+(?:\.\d{1,2})?)", raw, re.IGNORECASE)
                if m2:
                    val = m2.group(1).replace(',', '')
                    if len(re.sub(r"[^0-9]", "", val)) < 10:
                        return val
                # try processed text (converted words)
                amt = re.findall(r"([0-9,]+(?:\.\d{1,2})?)", proc)
                for a in amt:
                    if len(re.sub(r"[^0-9]", "", a)) < 10:
                        return a.replace(',', '')
                # fallback: convert words in raw and search
                try:
                    conv = convert_words_to_numbers(raw)
                    amt2 = re.findall(r"([0-9,]+(?:\.\d{1,2})?)", conv)
                    for a in amt2:
                        if len(re.sub(r"[^0-9]", "", a)) < 10:
                            return a.replace(',', '')
                except Exception:
                    pass
                return []

            # Names: prefer explicit intro phrases in raw text, else use NER on raw
            if label == "Name":
                # 1) explicit patterns: my name is / i am / this is / name[: ] (stop at punctuation or verbs)
                # NOTE: "this is" requires following word to start after mandatory non-article article
                name_pattern = re.search(r"(?:my name is|i am|this is|name[:\s])\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z'\-]{1,20})?)\b", raw, re.IGNORECASE)
                if name_pattern:
                    name_str = name_pattern.group(1).strip()
                    # filter out trailing verbs (and, to, deposit, etc)
                    stop_words = {"and", "to", "deposit", "transfer", "withdraw", "credit", "debit", "send", "move", "a", "the", "is"}
                    parts = name_str.split()
                    parts = [p for p in parts if p.lower() not in stop_words]
                    if parts and len(parts[0]) > 1:  # ensure first part is at least 2 chars (not single letter like "A")
                        return ' '.join(parts[:2]).title()  # max 2 parts (first + last name)
                    return []

                # 2) NER via spacy on raw text (if available), extract only first PERSON token
                global _SPACY_NLP
                try:
                    if _SPACY_NLP is None:
                        import spacy
                        _SPACY_NLP = spacy.load("en_core_web_sm")
                    doc = _SPACY_NLP(raw)
                    stop_tokens = {"and","i","want","to","deposit","transfer","withdraw","my","account","phone","credit","debit","send","move","please","is","am","this","name","also","but","where","a","the"}
                    for ent in doc.ents:
                        if ent.label_ == "PERSON":
                            parts = [t for t in ent.text.strip().split() if t.lower() not in stop_tokens]
                            if parts:
                                # only accept first 1-2 tokens as name
                                return ' '.join(parts[:2]).title()
                except Exception:
                    pass

                # 3) NO fallback: if no explicit intro phrase or PERSON NER, return nothing (don't guess)
                return []

            logging.info("Field Extraction Completed")
            return []
        except Exception as e:
            raise AssisstantException(e, sys)