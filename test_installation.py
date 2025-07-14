#!/usr/bin/env python3
"""
Test script to validate Reddit Persona Generator installation
"""

import sys
import subprocess

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'praw', 'requests', 'bs4', 'nltk', 'textblob', 
        'pandas', 'numpy', 'json', 'datetime', 'collections'
    ]

    failed_imports = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False

    return True

def test_nltk_data():
    """Test if NLTK data is available"""
    import nltk

    required_data = [
        'punkt', 'stopwords', 'averaged_perceptron_tagger', 'vader_lexicon'
    ]

    missing_data = []

    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
            print(f"✅ NLTK {data} data found")
        except LookupError:
            try:
                nltk.data.find(f'corpora/{data}')
                print(f"✅ NLTK {data} data found")
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{data}')
                    print(f"✅ NLTK {data} data found")
                except LookupError:
                    print(f"❌ NLTK {data} data not found")
                    missing_data.append(data)

    if missing_data:
        print(f"\n⚠️  Missing NLTK data: {', '.join(missing_data)}")
        print("Run the main script once to automatically download required data")
        return False

    return True

def test_reddit_persona_generator():
    """Test if the main script can be imported"""
    try:
        sys.path.insert(0, '.')
        import reddit_persona_generator
        print("✅ Reddit Persona Generator script imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import Reddit Persona Generator: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Reddit Persona Generator Installation\n")

    print("1. Testing package imports...")
    imports_ok = test_imports()
    print()

    print("2. Testing NLTK data...")
    nltk_ok = test_nltk_data()
    print()

    print("3. Testing main script...")
    script_ok = test_reddit_persona_generator()
    print()

    if imports_ok and nltk_ok and script_ok:
        print("🎉 All tests passed! The Reddit Persona Generator is ready to use.")
        print("\nTo run the script:")
        print("python reddit_persona_generator.py https://www.reddit.com/user/USERNAME")
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
