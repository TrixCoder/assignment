#!/usr/bin/env python3
"""
Example usage script for Reddit Persona Generator
"""

import os
import sys
from reddit_persona_generator import RedditPersonaGenerator

def main():
    """Example usage of the Reddit Persona Generator"""

    print("🚀 Reddit Persona Generator - Example Usage\n")

    # Example 1: Using without Reddit API credentials (web scraping only)
    print("Example 1: Web scraping mode (no API credentials)")
    print("-" * 50)

    generator = RedditPersonaGenerator()

    # Test with sample user URLs
    test_urls = [
        "https://www.reddit.com/user/kojied/",
        "https://www.reddit.com/user/Hungry-Move-6603/"
    ]

    for url in test_urls:
        try:
            print(f"Processing: {url}")
            username = generator.extract_username_from_url(url)
            print(f"Extracted username: {username}")

            # Note: This is just demonstrating the URL extraction
            # Full processing would require actual Reddit data
            print(f"Would generate: {username}_persona.txt")
            print()

        except Exception as e:
            print(f"Error processing {url}: {e}")
            print()

    print("\n📚 To run with actual data:")
    print("python reddit_persona_generator.py https://www.reddit.com/user/USERNAME")

    print("\n🔑 For better results, set up Reddit API credentials:")
    print("1. Edit reddit_config.json with your Reddit API credentials")
    print("2. Run: python reddit_persona_generator.py URL --config reddit_config.json")

    print("\n📄 Output files will be saved as: USERNAME_persona.txt")

    # Example 2: Show what the configuration should look like
    print("\nExample Reddit API Configuration (reddit_config.json):")
    print("-" * 55)
    config_example = """{
    "client_id": "your_client_id_here",
    "client_secret": "your_client_secret_here", 
    "user_agent": "PersonaGenerator/1.0 by YourUsername",
    "username": "your_reddit_username",
    "password": "your_reddit_password"
}"""
    print(config_example)

    print("\n🎯 Sample output files have been created:")
    print("- kojied_persona.txt")
    print("- Hungry-Move-6603_persona.txt")

    print("\n✅ Ready to generate personas! Use the main script with real Reddit user URLs.")

if __name__ == "__main__":
    main()
