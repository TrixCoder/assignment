#!/usr/bin/env python3
"""
Reddit User Persona Generator – Dynamic Content Extraction
Generates a professional infographic PNG with actual user data extracted from Reddit profiles.

Author: AI Assistant
Date: 2025-07-14
"""

import os
import re
import sys
import time
import json
import math
import random
import logging
import argparse
import requests
import threading
import numpy as np
from io import BytesIO
from datetime import datetime
from textblob import TextBlob
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Optional Reddit API
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

# -- Logging Setup ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PersonaGen")

# -- NLTK Download -----------------------------------------------------------
nltk_packages = [
    "punkt", "punkt_tab", "vader_lexicon",
    "stopwords", "wordnet", "omw-1.4"
]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# -- Data Classes -------------------------------------------------------------
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RedditPost:
    title: str
    content: str
    score: int
    subreddit: str
    created_utc: float
    url: str
    comment_count: int
    post_id: str

@dataclass
class RedditComment:
    content: str
    score: int
    subreddit: str
    created_utc: float
    comment_id: str
    parent_id: str

@dataclass
class UserPersona:
    username: str
    analysis_date: str
    account_age_days: int
    total_posts: int
    total_comments: int
    total_karma: int
    post_karma: int
    comment_karma: int
    posting_frequency: float
    comment_frequency: float
    average_post_score: float
    average_comment_score: float
    extraversion: float
    emotional_stability: float
    openness: float
    average_sentiment: float
    sentiment_polarity: float
    sentiment_subjectivity: float
    communication_style: str
    activity_level: str
    top_interests: List[str]
    top_subreddits: List[str]
    behavior_patterns: List[str]
    frustrations: List[str]
    goals_needs: List[str]
    motivations: List[Tuple[str, float]]
    user_quote: str

# -- Persona Generator --------------------------------------------------------
class RedditPersonaGenerator:
    def __init__(self, config_path: str = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PersonaGenerator/1.0 (Research Tool)"
        })
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.last_request = 0.0
        if config_path and PRAW_AVAILABLE:
            self._init_praw(config_path)
        else:
            self.reddit = None

    def _init_praw(self, cfg: str):
        try:
            data = json.load(open(cfg))
            self.reddit = praw.Reddit(
                client_id=data["client_id"],
                client_secret=data["client_secret"],
                user_agent=data["user_agent"],
                username=data.get("username"),
                password=data.get("password")
            )
            _ = self.reddit.user.me()
            logger.info("Authenticated via PRAW")
        except Exception as e:
            logger.warning(f"PRAW authentication failed: {e}")
            self.reddit = None

    def _rate_limit(self, interval: float = 1.5):
        dt = time.time() - self.last_request
        if dt < interval:
            time.sleep(interval - dt)
        self.last_request = time.time()

    def extract_username(self, url: str) -> str:
        u = url.strip().rstrip("/")
        patterns = [
            r"/user/([^/]+)", r"/u/([^/]+)",
            r"reddit\.com/user/([^/]+)", r"reddit\.com/u/([^/]+)"
        ]
        for p in patterns:
            m = re.search(p, u, re.IGNORECASE)
            if m: return m.group(1)
        return re.sub(r"[^A-Za-z0-9_\-]", "", u)

    def scrape_api(self, user: str) -> Tuple[List[RedditPost], List[RedditComment]]:
        if not self.reddit:
            return [], []
        posts, comments = [], []
        try:
            redditor = self.reddit.redditor(user)
            
            # Get user posts
            for sub in redditor.submissions.new(limit=100):
                posts.append(RedditPost(
                    title=sub.title or "",
                    content=sub.selftext or "",
                    score=sub.score or 0,
                    subreddit=str(sub.subreddit),
                    created_utc=sub.created_utc,
                    url=sub.url or "",
                    comment_count=sub.num_comments or 0,
                    post_id=sub.id
                ))
            
            # Get user comments
            for com in redditor.comments.new(limit=200):
                comments.append(RedditComment(
                    content=com.body or "",
                    score=com.score or 0,
                    subreddit=str(com.subreddit),
                    created_utc=com.created_utc,
                    comment_id=com.id,
                    parent_id=com.parent_id
                ))
                
        except Exception as e:
            logger.warning(f"PRAW scrape failed: {e}")
        return posts, comments

    def scrape_web(self, user: str) -> Tuple[List[RedditPost], List[RedditComment]]:
        posts, comments = [], []
        urls = [
            f"https://old.reddit.com/user/{user}",
            f"https://www.reddit.com/user/{user}"
        ]
        resp = None
        for u in urls:
            self._rate_limit(2.0)
            try:
                resp = self.session.get(u, timeout=15)
                if resp.status_code == 200:
                    logger.info(f"Successfully accessed {u}")
                    break
            except Exception as e:
                logger.warning(f"Failed to access {u}: {e}")
                continue
                
        if not resp or resp.status_code != 200:
            return posts, comments
            
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Parse posts
        for e in soup.select("div.thing.link")[:50]:
            t = e.select_one("a.title")
            s = e.select_one("a.subreddit")
            if t:
                posts.append(RedditPost(
                    title=t.get_text(strip=True),
                    content="",
                    score=self._extract_score(e),
                    subreddit=s.get_text(strip=True) if s else "unknown",
                    created_utc=time.time(),
                    url=t.get('href', ''),
                    comment_count=0,
                    post_id=e.get('data-fullname', '')
                ))
        
        # Parse comments
        for e in soup.select("div.thing.comment")[:100]:
            b = e.select_one("div.md")
            s = e.select_one("a.subreddit")
            if b:
                comments.append(RedditComment(
                    content=b.get_text(strip=True),
                    score=self._extract_score(e),
                    subreddit=s.get_text(strip=True) if s else "unknown",
                    created_utc=time.time(),
                    comment_id=e.get('data-fullname', ''),
                    parent_id=''
                ))
        
        logger.info(f"Web scraping completed: {len(posts)} posts, {len(comments)} comments")
        return posts, comments

    def _extract_score(self, element):
        """Extract score from Reddit element"""
        score_selectors = ['div.score', 'span.score', 'div.unvoted']
        for selector in score_selectors:
            score_elem = element.select_one(selector)
            if score_elem:
                score_text = score_elem.get_text(strip=True)
                try:
                    return int(re.search(r'(-?\d+)', score_text).group(1))
                except:
                    continue
        return 0

    def analyze_sentiment(self, text: str) -> Tuple[float, float, float]:
        if not text:
            return 0.0, 0.0, 0.0
        v = self.sentiment_analyzer.polarity_scores(text)
        tb = TextBlob(text)
        return v["compound"], tb.sentiment.polarity, tb.sentiment.subjectivity

    def extract_interests(self, posts, comments) -> List[str]:
        """Extract actual user interests from content"""
        txts = []
        for p in posts:
            if p.title: txts.append(p.title)
            if p.content: txts.append(p.content)
        for c in comments:
            if c.content: txts.append(c.content)
            
        if not txts:
            return ["No interests data available"]
            
        combined = " ".join(txts).lower()
        tokens = word_tokenize(combined)
        filtered = [
            self.lemmatizer.lemmatize(t) for t in tokens
            if t.isalpha() and len(t) > 3 and t not in self.stop_words
        ]
        
        # Get meaningful interests
        interests = [w for w, _ in Counter(filtered).most_common(15) if len(w) > 3]
        return interests[:8] if interests else ["No interests identified"]

    def extract_behavior_patterns(self, posts, comments) -> List[str]:
        """Extract actual behavior patterns from user activity"""
        patterns = []
        
        if posts:
            avg_post_length = np.mean([len(p.title + p.content) for p in posts])
            if avg_post_length > 200:
                patterns.append("Creates detailed, comprehensive posts")
            else:
                patterns.append("Prefers concise, brief posts")
                
            # Posting time patterns
            post_hours = [datetime.fromtimestamp(p.created_utc).hour for p in posts if p.created_utc]
            if post_hours:
                avg_hour = np.mean(post_hours)
                if 6 <= avg_hour <= 12:
                    patterns.append("Most active during morning hours")
                elif 12 <= avg_hour <= 18:
                    patterns.append("Most active during afternoon hours")
                else:
                    patterns.append("Most active during evening/night hours")
        
        if comments:
            avg_comment_length = np.mean([len(c.content) for c in comments])
            if avg_comment_length > 100:
                patterns.append("Writes detailed, thoughtful comments")
            else:
                patterns.append("Prefers short, quick responses")
                
            # Engagement patterns
            question_comments = sum(1 for c in comments if '?' in c.content)
            if question_comments > len(comments) * 0.3:
                patterns.append("Frequently asks questions and seeks information")
            
            helpful_keywords = ['thanks', 'help', 'appreciate', 'useful', 'great']
            helpful_comments = sum(1 for c in comments if any(kw in c.content.lower() for kw in helpful_keywords))
            if helpful_comments > len(comments) * 0.2:
                patterns.append("Often expresses gratitude and appreciation")
        
        # Subreddit diversity
        unique_subs = set()
        for p in posts: unique_subs.add(p.subreddit)
        for c in comments: unique_subs.add(c.subreddit)
        
        if len(unique_subs) > 10:
            patterns.append("Participates in diverse communities")
        elif len(unique_subs) > 5:
            patterns.append("Active in several focused communities")
        else:
            patterns.append("Focuses on specific communities")
            
        return patterns[:6] if patterns else ["Limited activity data available"]

    def extract_frustrations(self, posts, comments) -> List[str]:
        """Extract frustrations from user content"""
        frustrations = []
        
        # Negative sentiment indicators
        negative_keywords = ['annoying', 'frustrating', 'hate', 'terrible', 'awful', 'worst', 'problem', 'issue', 'broken', 'stupid']
        complaint_keywords = ['why does', 'why do', 'cant believe', 'so annoying', 'makes no sense']
        
        all_content = []
        for p in posts:
            all_content.append(p.title + " " + p.content)
        for c in comments:
            all_content.append(c.content)
        
        frustration_content = []
        for content in all_content:
            content_lower = content.lower()
            if any(kw in content_lower for kw in negative_keywords + complaint_keywords):
                # Extract the sentence containing frustration
                sentences = content.split('.')
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in negative_keywords + complaint_keywords):
                        frustration_content.append(sentence.strip())
                        break
        
        # Process and clean frustrations
        for frustration in frustration_content[:5]:
            if len(frustration) > 20 and len(frustration) < 150:
                clean_frustration = re.sub(r'[^\w\s]', '', frustration)
                if clean_frustration:
                    frustrations.append(f"• {clean_frustration.capitalize()}")
        
        if not frustrations:
            # Infer frustrations from activity patterns
            if len(comments) > len(posts) * 5:
                frustrations.append("• Spends more time commenting than creating original content")
            
            low_score_posts = [p for p in posts if p.score < 5]
            if len(low_score_posts) > len(posts) * 0.7:
                frustrations.append("• Posts often receive limited engagement")
                
            if not frustrations:
                frustrations.append("• No specific frustrations identified from available data")
        
        return frustrations[:5]

    def extract_goals_needs(self, posts, comments) -> List[str]:
        """Extract goals and needs from user content"""
        goals = []
        
        goal_keywords = ['want to', 'need to', 'trying to', 'hope to', 'plan to', 'goal', 'objective', 'looking for']
        help_keywords = ['help', 'advice', 'suggestion', 'recommendation', 'how to']
        
        all_content = []
        for p in posts:
            all_content.append(p.title + " " + p.content)
        for c in comments:
            all_content.append(c.content)
        
        goal_content = []
        for content in all_content:
            content_lower = content.lower()
            if any(kw in content_lower for kw in goal_keywords + help_keywords):
                sentences = content.split('.')
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in goal_keywords + help_keywords):
                        goal_content.append(sentence.strip())
                        break
        
        # Process goals
        for goal in goal_content[:4]:
            if len(goal) > 15 and len(goal) < 120:
                clean_goal = re.sub(r'[^\w\s]', '', goal)
                if clean_goal:
                    goals.append(f"• {clean_goal.capitalize()}")
        
        if not goals:
            # Infer goals from activity patterns
            top_subreddits = Counter()
            for p in posts: top_subreddits[p.subreddit] += 1
            for c in comments: top_subreddits[c.subreddit] += 1
            
            if top_subreddits:
                top_sub = top_subreddits.most_common(1)[0][0]
                goals.append(f"• Active engagement in {top_sub} community")
                goals.append(f"• Building knowledge and connections through Reddit")
            else:
                goals.append("• No specific goals identified from available data")
        
        return goals[:4]

    def calculate_motivations(self, posts, comments) -> List[Tuple[str, float]]:
        """Calculate user motivations based on activity"""
        motivations = {}
        
        # Information seeking
        question_ratio = sum(1 for c in comments if '?' in c.content) / max(len(comments), 1)
        motivations["INFORMATION"] = min(1.0, question_ratio * 3)
        
        # Social interaction
        social_keywords = ['thanks', 'great', 'awesome', 'agree', 'disagree', 'think', 'feel']
        social_comments = sum(1 for c in comments if any(kw in c.content.lower() for kw in social_keywords))
        motivations["SOCIAL"] = min(1.0, social_comments / max(len(comments), 1) * 2)
        
        # Knowledge sharing
        long_posts = sum(1 for p in posts if len(p.content) > 200)
        motivations["SHARING"] = min(1.0, long_posts / max(len(posts), 1) * 2)
        
        # Entertainment
        entertainment_subs = ['funny', 'memes', 'gaming', 'movies', 'tv', 'music']
        entertainment_activity = sum(1 for p in posts if any(sub in p.subreddit.lower() for sub in entertainment_subs))
        entertainment_activity += sum(1 for c in comments if any(sub in c.subreddit.lower() for sub in entertainment_subs))
        motivations["ENTERTAINMENT"] = min(1.0, entertainment_activity / max(len(posts) + len(comments), 1) * 3)
        
        # Community building
        unique_subs = len(set([p.subreddit for p in posts] + [c.subreddit for c in comments]))
        motivations["COMMUNITY"] = min(1.0, unique_subs / 20)
        
        # Support seeking - FIXED VERSION
        support_keywords = ['help', 'advice', 'support', 'problem', 'issue']
        
        # Handle posts (have title and content)
        support_posts = sum(1 for p in posts if any(kw in (p.title + " " + p.content).lower() for kw in support_keywords))
        
        # Handle comments (only have content)
        support_comments = sum(1 for c in comments if any(kw in c.content.lower() for kw in support_keywords))
        
        support_content = support_posts + support_comments
        motivations["SUPPORT"] = min(1.0, support_content / max(len(posts) + len(comments), 1) * 4)
        
        return [(k, v) for k, v in sorted(motivations.items(), key=lambda x: x[1], reverse=True)]


    def generate_user_quote(self, posts, comments) -> str:
        """Generate a representative quote from user's actual content"""
        # Look for meaningful quotes from comments
        potential_quotes = []
        
        for comment in comments:
            content = comment.content.strip()
            # Look for first-person statements
            if any(phrase in content.lower() for phrase in ['i think', 'i believe', 'i feel', 'i want', 'i need', 'i love', 'i hate']):
                if 20 < len(content) < 150 and '?' not in content:
                    potential_quotes.append(content)
        
        if potential_quotes:
            # Select the most representative quote
            return f'"{potential_quotes[0]}"'
        
        # Fallback to post titles
        for post in posts:
            if 20 < len(post.title) < 100:
                return f'"{post.title}"'
        
        # Generic fallback based on activity
        if len(comments) > len(posts):
            return '"I enjoy engaging in discussions and sharing my thoughts with the community."'
        else:
            return '"I prefer to share content and contribute to the conversation."'

    def compute_persona(self, user: str) -> UserPersona:
        """Generate complete user persona with real extracted data"""
        logger.info(f"Processing user: {user}")
        
        posts, comments = self.scrape_api(user)
        if not posts and not comments:
            logger.info("API data insufficient, trying web scraping...")
            posts, comments = self.scrape_web(user)

        date = datetime.now().isoformat()
        tp, tc = len(posts), len(comments)
        total_karma = sum(p.score for p in posts) + sum(c.score for c in comments)
        post_karma = sum(p.score for p in posts)
        comment_karma = sum(c.score for c in comments)
        
        pf = tp / 30.0; cf = tc / 30.0
        avg_ps = np.mean([p.score for p in posts]) if posts else 0.0
        avg_cs = np.mean([c.score for c in comments]) if comments else 0.0

        # Sentiment analysis
        comps, pols, subs = [], [], []
        for p in posts:
            c_, pol_, sub_ = self.analyze_sentiment(p.title + " " + p.content)
            comps.append(c_); pols.append(pol_); subs.append(sub_)
        for c in comments:
            c_, pol_, sub_ = self.analyze_sentiment(c.content)
            comps.append(c_); pols.append(pol_); subs.append(sub_)

        avg_comp = float(np.nanmean(comps)) if comps else 0.0
        avg_pol = float(np.nanmean(pols)) if pols else 0.0
        avg_sub = float(np.nanmean(subs)) if subs else 0.0

        # Personality traits
        extr = min(1.0, 0.3 + (len(comments) / max(len(posts) + len(comments), 1)))
        stab = max(0.0, min(1.0, 0.7 + avg_comp))
        opp = min(1.0, 0.2 + len(set([p.subreddit for p in posts] + [c.subreddit for c in comments])) * 0.05)

        # Communication style
        if comments:
            avg_comment_length = np.mean([len(c.content) for c in comments])
            comm_style = "verbose" if avg_comment_length > 100 else "concise"
        else:
            comm_style = "minimal"

        # Activity level
        total_activity = tp + tc
        if total_activity > 100:
            act_level = "high"
        elif total_activity > 30:
            act_level = "moderate"
        else:
            act_level = "low"

        # Extract dynamic content
        interests = self.extract_interests(posts, comments)
        subreddits = [s for s, _ in Counter(
            [p.subreddit for p in posts] + [c.subreddit for c in comments]
        ).most_common(8)]
        
        behavior_patterns = self.extract_behavior_patterns(posts, comments)
        frustrations = self.extract_frustrations(posts, comments)
        goals_needs = self.extract_goals_needs(posts, comments)
        motivations = self.calculate_motivations(posts, comments)
        user_quote = self.generate_user_quote(posts, comments)

        return UserPersona(
            username=user,
            analysis_date=date,
            account_age_days=365,  # Would need API for real data
            total_posts=tp,
            total_comments=tc,
            total_karma=total_karma,
            post_karma=post_karma,
            comment_karma=comment_karma,
            posting_frequency=pf,
            comment_frequency=cf,
            average_post_score=avg_ps,
            average_comment_score=avg_cs,
            extraversion=extr,
            emotional_stability=stab,
            openness=opp,
            average_sentiment=avg_comp,
            sentiment_polarity=avg_pol,
            sentiment_subjectivity=avg_sub,
            communication_style=comm_style,
            activity_level=act_level,
            top_interests=interests,
            top_subreddits=subreddits,
            behavior_patterns=behavior_patterns,
            frustrations=frustrations,
            goals_needs=goals_needs,
            motivations=motivations,
            user_quote=user_quote
        )

    def render_png(self, persona: UserPersona, out_path: str):
        """Create a professional persona infographic with real user data."""
        # Canvas dimensions matching reference
        width, height = 1152, 1536
        
        # Color scheme
        bg_color = "#f8f9fa"
        header_color = "#ff4500"
        panel_color = "#ffffff"
        text_color = "#1a1a1b"
        
        # Create base image
        base = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(base)
        
        # Load fonts with fallback
        try:
            font_header = ImageFont.truetype("arialbd.ttf", 48)
            font_title = ImageFont.truetype("arialbd.ttf", 32)
            font_body = ImageFont.truetype("arial.ttf", 20)
            font_small = ImageFont.truetype("arial.ttf", 18)
            font_date = ImageFont.truetype("arial.ttf", 24)
        except:
            font_header = ImageFont.load_default()
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_date = ImageFont.load_default()
        
        # Header section
        header_height = 192
        draw.rectangle([0, 0, width, header_height], fill=header_color)
        
        # Username
        draw.text((50, 60), f"u/{persona.username}", fill="white", font=font_header)
        
        # Date (top right)
        date_text = persona.analysis_date.split("T")[0]
        draw.text((width - 200, 60), date_text, fill="white", font=font_date)
        
        # Section positions
        current_y = header_height + 40
        panel_margin = 48
        text_margin = 64
        
        # Helper function to draw section
        def draw_section(title, y_pos):
            draw.rectangle([panel_margin, y_pos, width - panel_margin, y_pos + 40], 
                          fill=panel_color, outline="#ddd", width=1)
            draw.text((text_margin, y_pos + 8), title, fill=text_color, font=font_title)
            return y_pos + 40 + 16
        
        # Behavior & Habits
        content_y = draw_section("BEHAVIOUR & HABITS", current_y)
        for i, pattern in enumerate(persona.behavior_patterns):
            draw.text((text_margin, content_y + i * 24), f"• {pattern}", fill=text_color, font=font_body)
        
        current_y = content_y + len(persona.behavior_patterns) * 24 + 40
        
        # Frustrations
        content_y = draw_section("FRUSTRATIONS", current_y)
        for i, frustration in enumerate(persona.frustrations):
            # Handle text wrapping for long frustrations
            if len(frustration) > 60:
                words = frustration.split()
                line1 = " ".join(words[:8])
                line2 = " ".join(words[8:])
                draw.text((text_margin, content_y + i * 48), line1, fill=text_color, font=font_body)
                if line2:
                    draw.text((text_margin + 20, content_y + i * 48 + 24), line2, fill=text_color, font=font_body)
            else:
                draw.text((text_margin, content_y + i * 24), frustration, fill=text_color, font=font_body)
        
        current_y = content_y + len(persona.frustrations) * 48 + 40
        
        # Goals & Needs
        content_y = draw_section("GOALS & NEEDS", current_y)
        for i, goal in enumerate(persona.goals_needs):
            # Handle text wrapping
            if len(goal) > 60:
                words = goal.split()
                line1 = " ".join(words[:8])
                line2 = " ".join(words[8:])
                draw.text((text_margin, content_y + i * 48), line1, fill=text_color, font=font_body)
                if line2:
                    draw.text((text_margin + 20, content_y + i * 48 + 24), line2, fill=text_color, font=font_body)
            else:
                draw.text((text_margin, content_y + i * 24), goal, fill=text_color, font=font_body)
        
        current_y = content_y + len(persona.goals_needs) * 48 + 40
        
        # Personality section
        content_y = draw_section("PERSONALITY", current_y)
        
        # Personality traits with sliders
        traits = [
            ("INTROVERT", "EXTROVERT", persona.extraversion),
            ("INTUITION", "SENSING", 0.7),
            ("FEELING", "THINKING", persona.emotional_stability),
            ("PERCEIVING", "JUDGING", persona.openness)
        ]
        
        slider_y = content_y + 20
        slider_width = 400
        slider_height = 8
        
        for i, (left_label, right_label, value) in enumerate(traits):
            y_pos = slider_y + i * 40
            
            # Left label
            draw.text((text_margin, y_pos - 10), left_label, fill=text_color, font=font_small)
            
            # Right label
            right_x = text_margin + slider_width + 40
            draw.text((right_x, y_pos - 10), right_label, fill=text_color, font=font_small)
            
            # Slider background
            slider_x = text_margin + 120
            draw.rectangle([slider_x, y_pos, slider_x + slider_width, y_pos + slider_height], 
                          fill="#e0e0e0")
            
            # Slider indicator
            indicator_x = slider_x + (slider_width * value) - 4
            draw.rectangle([indicator_x, y_pos - 4, indicator_x + 8, y_pos + slider_height + 4], 
                          fill="#333333")
        
        current_y = slider_y + len(traits) * 40 + 40
        
        # Motivations section
        content_y = draw_section("MOTIVATIONS", current_y)
        
        bar_width = 300
        bar_height = 20
        
        for i, (label, value) in enumerate(persona.motivations[:6]):
            y_pos = content_y + i * 35
            
            # Label
            draw.text((text_margin, y_pos), label, fill=text_color, font=font_small)
            
            # Bar background
            bar_x = text_margin + 150
            draw.rectangle([bar_x, y_pos + 5, bar_x + bar_width, y_pos + 5 + bar_height], 
                          fill="#e0e0e0")
            
            # Bar fill
            fill_width = bar_width * value
            draw.rectangle([bar_x, y_pos + 5, bar_x + fill_width, y_pos + 5 + bar_height], 
                          fill="#ff4500")
        
        # Quote section
        quote_y = height - 200
        quote_bg_height = 120
        
        # Quote background
        draw.rectangle([0, quote_y, width, quote_y + quote_bg_height], fill="#ff4500")
        
        # Quote text - handle long quotes
        quote_text = persona.user_quote
        if len(quote_text) > 80:
            # Split long quotes into two lines
            words = quote_text.split()
            mid = len(words) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
            
            # Center both lines
            bbox1 = draw.textbbox((0, 0), line1, font=font_body)
            text_width1 = bbox1[2] - bbox1[0]
            text_x1 = (width - text_width1) // 2
            
            bbox2 = draw.textbbox((0, 0), line2, font=font_body)
            text_width2 = bbox2[2] - bbox2[0]
            text_x2 = (width - text_width2) // 2
            
            draw.text((text_x1, quote_y + 25), line1, fill="white", font=font_body)
            draw.text((text_x2, quote_y + 55), line2, fill="white", font=font_body)
        else:
            # Single line quote
            bbox = draw.textbbox((0, 0), quote_text, font=font_body)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            draw.text((text_x, quote_y + 40), quote_text, fill="white", font=font_body)
        
        # Save the image
        base.save(out_path, format="PNG", dpi=(300, 300))
        logger.info(f"Saved professional infographic: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Reddit Persona → PNG Generator")
    p.add_argument("url", help="Reddit profile URL or username")
    p.add_argument("--config", help="reddit_config.json for PRAW", default=None)
    p.add_argument("--output", help="Output directory", default=".")
    args = p.parse_args()

    gen = RedditPersonaGenerator(config_path=args.config)
    user = gen.extract_username(args.url)
    persona = gen.compute_persona(user)

    os.makedirs(args.output, exist_ok=True)
    out_png = os.path.join(args.output, f"{user}_persona.png")
    gen.render_png(persona, out_png)
    print(f"✅ Generated: {out_png}")

if __name__ == "__main__":
    main()
