from assisstants.processor.text_processor import TextProcessor
from assisstants.Classifier.text_classifier import TextClassifier
from assisstants.extractor.fields_extractor import ExtractFields

p = TextProcessor()
c = TextClassifier()
e = ExtractFields()

text = "where are you going"
processed = p.process_text(text)
try:
    lbl, conf = c.classify(processed, return_prob=True)
except TypeError:
    lbl = c.classify(processed)
    conf = None

# Try extractor for Name heuristics
entity = e.extract('Name', text)

out = []
out.append(f"processed: {processed}")
out.append(f"classifier: {lbl} {conf}")
out.append(f"extractor(Name): {entity}")
with open('run_quick_test_out.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out))
print('Wrote run_quick_test_out.txt')
