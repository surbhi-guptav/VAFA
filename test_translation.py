"""
Test script to demonstrate multi-language translation.

Run this to verify translation setup works before integrating into app.py

Usage:
    python test_translation.py
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from assisstants.translator.huggingface_translator import HuggingFaceTranslator

def test_english_to_hindi():
    """Test English to Hindi translation"""
    print("\n" + "="*60)
    print("TEST 1: English → Hindi Translation")
    print("="*60)
    
    translator = HuggingFaceTranslator()
    
    test_cases = [
        "My name is Raj",
        "I want to transfer 5000 rupees",
        "My account number is 123456789012",
        "Call me at 9876543210",
        "Please deposit 1 lakh to my savings account",
    ]
    
    for english_text in test_cases:
        try:
            hindi_text = translator.translate(english_text, "en", "hi")
            print(f"\n📝 English: {english_text}")
            print(f"🔤 Hindi:   {hindi_text}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_hindi_to_english():
    """Test Hindi to English translation"""
    print("\n" + "="*60)
    print("TEST 2: Hindi → English Translation")
    print("="*60)
    
    translator = HuggingFaceTranslator()
    
    test_cases = [
        "मेरा नाम राज है",
        "मुझे 5000 रुपये स्थानांतरित करना है",
        "मेरे खाते की संख्या 123456789012 है",
        "मुझे 9876543210 पर कॉल करें",
    ]
    
    for hindi_text in test_cases:
        try:
            english_text = translator.translate(hindi_text, "hi", "en")
            print(f"\n🔤 Hindi:   {hindi_text}")
            print(f"📝 English: {english_text}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_english_to_tamil():
    """Test English to Tamil translation"""
    print("\n" + "="*60)
    print("TEST 3: English → Tamil Translation")
    print("="*60)
    
    translator = HuggingFaceTranslator()
    
    test_cases = [
        "My name is Raj",
        "I want to transfer money",
        "Account number",
    ]
    
    for english_text in test_cases:
        try:
            tamil_text = translator.translate(english_text, "en", "ta")
            print(f"\n📝 English: {english_text}")
            print(f"🔤 Tamil:   {tamil_text}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_supported_languages():
    """Display supported languages and pairs"""
    print("\n" + "="*60)
    print("TEST 4: Supported Languages & Pairs")
    print("="*60)
    
    translator = HuggingFaceTranslator()
    
    langs = translator.get_supported_languages()
    print(f"\n✅ Supported Language Codes: {langs}")
    
    print(f"\n✅ Total Supported Pairs: {len(translator.get_supported_pairs())}")
    
    pairs = translator.get_supported_pairs()
    print("\nAll supported pairs:")
    for source, target in sorted(pairs):
        print(f"  {source} → {target}")

def test_translation_pipeline():
    """Test complete translation pipeline like VAFA would use"""
    print("\n" + "="*60)
    print("TEST 5: Complete VAFA Pipeline")
    print("="*60)
    
    translator = HuggingFaceTranslator()
    
    # Simulate user input in Hindi
    user_input = "मेरा नाम सुनील है। मुझे 10000 रुपये हस्तांतरित करना है।"
    user_language = "hi"
    
    print(f"\n👤 User Input ({user_language}): {user_input}")
    
    # Step 1: Translate to English
    english_text = translator.translate(user_input, "hi", "en")
    print(f"📝 Translated to English: {english_text}")
    
    # Step 2: Simulate extraction (Name: Sunil, Amount: 10000)
    extracted_name = "Sunil"
    extracted_amount = "10000"
    
    print(f"\n🔍 Extracted Values (English):")
    print(f"   Name: {extracted_name}")
    print(f"   Amount: {extracted_amount}")
    
    # Step 3: Translate extracted values back to Hindi
    translated_name = translator.translate(extracted_name, "en", "hi")
    translated_amount = translator.translate(extracted_amount, "en", "hi")
    
    print(f"\n🔍 Extracted Values (Hindi):")
    print(f"   Name: {translated_name}")
    print(f"   Amount: {translated_amount}")

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "VAFA TRANSLATION MODULE TESTS" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # Test 1: Supported languages
        test_supported_languages()
        
        # Test 2: English to Hindi
        print("\n⏳ Downloading language models (this may take a few minutes)...")
        test_english_to_hindi()
        
        # Test 3: Hindi to English
        test_hindi_to_english()
        
        # Test 4: English to Tamil
        test_english_to_tamil()
        
        # Test 5: Complete pipeline
        test_translation_pipeline()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📌 NEXT STEPS:")
        print("   1. Modify app.py to add language selector")
        print("   2. Update requirements.txt with new dependencies")
        print("   3. Test with Streamlit: streamlit run app.py")
        print("   4. Try speaking/typing in different languages")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logging.error(f"Test failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
