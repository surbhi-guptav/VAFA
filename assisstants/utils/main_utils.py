import sys
from assisstants.exception.exception import AssisstantException

import re
from word2number import w2n

def _safe_w2n(phrase: str):
    try:
        return w2n.word_to_num(phrase)
    except Exception:
        return None


def convert_words_to_numbers(text):
    try:
        s = text
        # normalize commas
        s = re.sub(r",", "", s)

        # handle numeric abbreviations like '5k' or '5K' -> 5000
        s = re.sub(r"\b(\d+(?:\.\d+)?)\s*[kK]\b", lambda m: str(int(float(m.group(1)) * 1000)), s)

        # handle patterns like 'two lakh', 'five lakh' -> multiply by 100000
        def _lakh_repl(m):
            num = _safe_w2n(m.group(1))
            if num is None:
                return m.group(0)
            return str(int(num * 100000))

        s = re.sub(r"\b([a-zA-Z\s-]+?)\s+lakh\b", _lakh_repl, s, flags=re.IGNORECASE)

        def _crore_repl(m):
            num = _safe_w2n(m.group(1))
            if num is None:
                return m.group(0)
            return str(int(num * 10000000))

        s = re.sub(r"\b([a-zA-Z\s-]+?)\s+crore\b", _crore_repl, s, flags=re.IGNORECASE)

        # handle 'thousand' explicitly
        def _thousand_repl(m):
            num = _safe_w2n(m.group(1))
            if num is None:
                return m.group(0)
            return str(int(num * 1000))

        s = re.sub(r"\b([a-zA-Z\s-]+?)\s+thousand\b", _thousand_repl, s, flags=re.IGNORECASE)

        # attempt to convert remaining pure word-number phrases
        # find sequences of only alphabetic words (up to 6 words) that look numeric
        def _wordnum_repl(m):
            phrase = m.group(0)
            num = _safe_w2n(phrase)
            return str(num) if num is not None else phrase

        word_pattern = r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|crore|and|k|m|b|million|billion)(?:[\s-]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|crore|and|k|m|b|million|billion)){0,5}\b"
        s = re.sub(word_pattern, _wordnum_repl, s, flags=re.IGNORECASE)

        return s
    except Exception as e:
        raise AssisstantException(e, sys)

