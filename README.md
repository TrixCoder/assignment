# Reddit User Persona Generator

A comprehensive Python script that extracts Reddit user data and generates detailed user personas with proper citations and evidence tracking.

## 🎯 Features

- **Dual Data Collection**: Supports both Reddit API (PRAW) and web scraping approaches
- **Comprehensive Analysis**: Performs sentiment analysis, topic modeling, and personality trait detection
- **Citation System**: Automatically tracks and cites specific posts/comments that support each persona characteristic
- **Multiple Output Formats**: Generates detailed text files with structured persona information
- **Rate Limiting**: Implements proper request throttling to respect Reddit's API guidelines
- **Error Handling**: Robust error handling with fallback mechanisms

## 📋 Requirements

- Python 3.7+
- Reddit API credentials (optional but recommended)
- Internet connection

## 🚀 Installation

1. **Clone or download the project files**
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (will be done automatically on first run):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('vader_lexicon')
   ```

## 🔧 Setup

### Option 1: Using Reddit API (Recommended)

1. **Create a Reddit App:**
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Choose "script" as the app type
   - Note down your `client_id` and `client_secret`

2. **Configure API credentials:**
   - Edit `reddit_config.json` with your credentials:
   ```json
   {
       "client_id": "your_client_id_here",
       "client_secret": "your_client_secret_here",
       "user_agent": "PersonaGenerator/1.0 by YourUsername",
       "username": "your_reddit_username",
       "password": "your_reddit_password"
   }
   ```

### Option 2: Web Scraping Only

If you don't have Reddit API credentials, the script will automatically fall back to web scraping mode.

## 📖 Usage

### Basic Usage

```bash
python reddit_persona_generator.py https://www.reddit.com/user/kojied/
```

### With Custom Configuration

```bash
python reddit_persona_generator.py https://www.reddit.com/user/kojied/ --config reddit_config.json
```

### With Custom Output Directory

```bash
python reddit_persona_generator.py https://www.reddit.com/user/kojied/ --output ./personas/
```

### Command Line Arguments

- `user_url`: Reddit user profile URL (required)
- `--config`: Path to Reddit API configuration file (optional)
- `--output`: Output directory for persona files (default: current directory)

## 📊 Output Format

The script generates a comprehensive persona text file with the following sections:

### 1. Basic Information
- Username
- Account age
- Karma scores (comment and link)

### 2. Personality Traits
- Extraversion level
- Emotional stability
- Openness to experience

### 3. Interests and Topics
- Most frequently mentioned topics
- Interest areas with frequency counts

### 4. Behavioral Patterns
- Activity level
- Posting frequency
- Primary subreddits

### 5. Communication Style
- Writing style description
- Average post length
- Usage of questions and exclamations

### 6. Sentiment Profile
- Overall polarity (positive/negative)
- Subjectivity level
- Compound sentiment score

### 7. Engagement Metrics
- Average submission and comment scores
- Post-to-comment ratio
- Total activity counts

### 8. Citations and Evidence
- Specific posts/comments supporting each characteristic
- Confidence levels for each trait
- Direct links to original content

## 🔍 Example Output

```
REDDIT USER PERSONA: KOJIED
==================================================

BASIC INFORMATION:
- Username: kojied
- Account Age: 1,234 days
- Comment Karma: 5,678
- Link Karma: 1,234

PERSONALITY TRAITS:
- Extraversion: high
- Emotional Stability: moderate
- Openness: high

INTERESTS AND TOPICS:
- Programming (mentioned 45 times)
- Technology (mentioned 32 times)
- Gaming (mentioned 28 times)
...

CITATIONS AND EVIDENCE:
=========================

Citation 1: Interest in Programming
Confidence: high
Evidence:
  1. Submission: "Just finished my first Python project..."
     URL: https://reddit.com/r/Python/comments/...
  2. Comment: "I love working with APIs because..."
     URL: https://reddit.com/r/webdev/comments/...
```

## 🛠️ Technical Implementation

### Data Collection Methods

1. **Reddit API (PRAW)**:
   - Retrieves up to 100 most recent submissions
   - Retrieves up to 200 most recent comments
   - Includes metadata like scores, timestamps, subreddits

2. **Web Scraping (BeautifulSoup)**:
   - Fallback method when API is unavailable
   - Scrapes user profile pages
   - Extracts post titles, comments, and basic metadata

### Analysis Components

1. **Sentiment Analysis**:
   - TextBlob for polarity and subjectivity
   - NLTK VADER for compound sentiment scoring

2. **Topic Extraction**:
   - NLTK tokenization and POS tagging
   - Frequency analysis of nouns and topics

3. **Personality Inference**:
   - Activity level analysis
   - Sentiment-based trait estimation
   - Communication pattern analysis

### Citation Generation

- Automatically finds posts/comments that support each persona characteristic
- Provides confidence levels based on evidence quantity
- Includes direct links to original Reddit content

## 🚨 Important Notes

### Rate Limiting
- The script implements rate limiting to respect Reddit's API guidelines
- Web scraping includes delays between requests
- Large profiles may take several minutes to process

### Privacy and Ethics
- Only processes publicly available Reddit data
- Respects Reddit's terms of service
- Users should obtain proper permissions before analyzing others' profiles

### Limitations
- Analysis quality depends on available post history
- Web scraping may be less comprehensive than API access
- Personality traits are inferred, not definitively measured

## 🔧 Troubleshooting

### Common Issues

1. **"Could not extract username from URL"**
   - Ensure the URL is in the format: `https://www.reddit.com/user/username`
   - Try both `/user/` and `/u/` formats

2. **"Reddit API not available"**
   - Check your `reddit_config.json` file
   - Verify your Reddit API credentials
   - The script will fall back to web scraping

3. **"Could not extract data for user"**
   - User profile might be private or deleted
   - Try a different user profile
   - Check internet connection

4. **NLTK download errors**
   - Run the script once to automatically download required NLTK data
   - Manually download: `python -c "import nltk; nltk.download('all')"`

### Performance Tips

- Use Reddit API credentials for better data quality
- Process users with substantial post history for better personas
- Run the script in a virtual environment to avoid conflicts

## 📚 Dependencies

- **praw**: Reddit API wrapper
- **requests**: HTTP library for web scraping
- **beautifulsoup4**: HTML parsing
- **nltk**: Natural language processing
- **textblob**: Sentiment analysis
- **pandas**: Data manipulation
- **numpy**: Numerical operations

## 🤝 Contributing

Feel free to contribute improvements:
- Enhanced personality trait detection
- Additional sentiment analysis methods
- Better citation algorithms
- Performance optimizations

## 📄 License

This project is for educational and research purposes. Please respect Reddit's terms of service and user privacy.

## 🔗 Sample Usage

Try the script with these example Reddit users:
- `https://www.reddit.com/user/kojied/`
- `https://www.reddit.com/user/Hungry-Move-6603/`

Remember to respect user privacy and Reddit's terms of service when using this tool.
