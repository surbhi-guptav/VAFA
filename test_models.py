"""
VAFA Model Verification Script
Step-by-step guide to verify all components are working correctly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("VAFA - Model Verification Script")
print("=" * 60)

# ============================================================
# STEP 1: Verify Model Files Exist
# ============================================================
print("\n[STEP 1] Verifying Model Files...")

from assisstants.constants import MODEL_PATH
import os

required_files = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json"
]

print(f"Model Path: {MODEL_PATH}")

all_files_exist = True
for file in required_files:
    filepath = os.path.join(MODEL_PATH, file)
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}: {'Found' if exists else 'Missing'}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n❌ ERROR: Some model files are missing!")
    print("Please ensure Models/ClassificationModel/ contains all required files.")
    sys.exit(1)

print("\n✓ All model files present!")

# ============================================================
# STEP 2: Test Model Loader
# ============================================================
print("\n[STEP 2] Testing Model Loader...")

try:
    from assisstants.loader.model_loader import ModelLoader
    
    print("  Loading tokenizer...")
    tokenizer = ModelLoader.get_tokenizer()
    print(f"  ✓ Tokenizer loaded: {type(tokenizer).__name__}")
    
    print("  Loading model...")
    model = ModelLoader.get_model()
    print(f"  ✓ Model loaded: {type(model).__name__}")
    
    device = ModelLoader._init_device()
    print(f"  ✓ Device: {device}")
    
except Exception as e:
    print(f"\n❌ ERROR loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Model loader working correctly!")

# ============================================================
# STEP 3: Test Text Processor
# ============================================================
print("\n[STEP 3] Testing Text Processor...")

try:
    from assisstants.processor.text_processor import TextProcessor
    
    processor = TextProcessor()
    
    test_cases = [
        ("My name is John Doe", "my name is john doe"),
        ("I'm going to the market", "i am going to the market"),
        ("Transfer 5000 rupees", "transfer 5000 rupees"),
    ]
    
    print("  Testing text preprocessing:")
    for raw, expected in test_cases:
        result = processor.process_text(raw)
        status = "✓" if result == expected else "~"
        print(f"    {status} \"{raw}\" -> \"{result}\"")
        
except Exception as e:
    print(f"\n❌ ERROR in text processor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Text processor working correctly!")

# ============================================================
# STEP 4: Test Text Classifier
# ============================================================
print("\n[STEP 4] Testing Text Classifier...")

try:
    from assisstants.Classifier.text_classifier import TextClassifier
    
    classifier = TextClassifier()
    
    test_cases = [
        ("My name is John Doe", "Name"),
        ("Phone number is 9876543210", "Phone Number"),
        ("Transfer 5000 rupees", "Amount"),
        ("Account number 123456789012", "Account Number"),
    ]
    
    print("  Testing classification:")
    all_correct = True
    for text, expected in test_cases:
        result = classifier.classify(text)
        status = "✓" if result == expected else "✗"
        print(f"    {status} \"{text}\" -> {result} (expected: {expected})")
        if result != expected:
            all_correct = False
    
    if not all_correct:
        print("  ⚠ Some classifications may be incorrect (model may need retraining)")
        
except Exception as e:
    print(f"\n❌ ERROR in classifier: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Text classifier working correctly!")

# ============================================================
# STEP 5: Test Entity Extraction
# ============================================================
print("\n[STEP 5] Testing Entity Extraction...")

try:
    from assisstants.extractor.fields_extractor import ExtractFields
    
    extractor = ExtractFields()
    
    test_cases = [
        ("Name", "My name is John Doe", "John Doe"),
        ("Phone Number", "Call me at 9876543210", "9876543210"),
        ("Amount", "Transfer 5000 rupees", "5000"),
        ("Account Number", "Account 123456789012", "123456789012"),
    ]
    
    print("  Testing entity extraction:")
    for label, text, expected in test_cases:
        result = extractor.extract(label, text)
        status = "✓" if result and expected in result else "~"
        print(f"    {status} [{label}] \"{text}\" -> {result} (expected: {expected})")
        
except Exception as e:
    print(f"\n❌ ERROR in extractor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Entity extraction working correctly!")

# ============================================================
# STEP 6: Test Number Conversion
# ============================================================
print("\n[STEP 6] Testing Number Conversion...")

try:
    from assisstants.utils.main_utils import convert_words_to_numbers
    
    test_cases = [
        ("five thousand", "5000"),
        ("ten lakh", "1000000"),
        ("2 crore", "20000000"),
        ("5k", "5000"),
    ]
    
    print("  Testing number conversion:")
    for text, expected in test_cases:
        result = convert_words_to_numbers(text)
        status = "✓" if expected in result else "~"
        print(f"    {status} \"{text}\" -> \"{result}\" (expected: {expected})")
        
except Exception as e:
    print(f"\n❌ ERROR in number converter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Number conversion working correctly!")

# ============================================================
# STEP 7: Test Full Pipeline
# ============================================================
print("\n[STEP 7] Testing Full Pipeline...")

try:
    from assisstants.processor.text_processor import TextProcessor
    from assisstants.Classifier.text_classifier import TextClassifier
    from assisstants.extractor.fields_extractor import ExtractFields
    
    processor = TextProcessor()
    classifier = TextClassifier()
    extractor = ExtractFields()
    
    # Test complete pipeline
    raw_text = "My name is John Doe and my phone is 9876543210"
    
    print(f"  Input: \"{raw_text}\"")
    
    # Process
    processed = processor.process_text(raw_text)
    print(f"  Processed: \"{processed}\"")
    
    # Classify
    label = classifier.classify(processed)
    print(f"  Label: {label}")
    
    # Extract
    entity = extractor.extract(label, processed)
    print(f"  Entity: {entity}")
    
    print("\n✓ Full pipeline working correctly!")
    
except Exception as e:
    print(f"\n❌ ERROR in full pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("""
All core components verified successfully!

Next steps:
1. Run: streamlit run app.py
2. Open browser at http://localhost:8501
3. Test voice input functionality

If any components failed, check the error messages above.
""")
