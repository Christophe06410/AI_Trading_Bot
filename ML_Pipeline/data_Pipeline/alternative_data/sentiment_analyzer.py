"""
Sentiment analysis for news, social media, and on-chain metrics
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
import re
from textblob import TextBlob
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sentiment data sources"""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    ONCHAIN = "onchain"


@dataclass
class SentimentConfig:
    """Sentiment analyzer configuration"""
    news_api_key: str = ""
    twitter_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    coingecko_api_key: str = ""
    fetch_interval_minutes: int = 5
    cache_ttl_hours: int = 24
    min_confidence: float = 0.3


class SentimentAnalyzer:
    """Analyzes sentiment from multiple sources for crypto assets"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Keywords for crypto sentiment
        self.crypto_keywords = {
            "positive": [
                "bullish", "moon", "pump", "breakout", "support", "accumulate",
                "buy", "long", "rally", "surge", "green", "gains", "recovery"
            ],
            "negative": [
                "bearish", "dump", "crash", "sell", "short", "resistance",
                "rejection", "fud", "scam", "warning", "red", "loss", "correction"
            ],
            "neutral": [
                "consolidation", "range", "sideways", "volatile", "uncertain",
                "waiting", "watching", "monitoring", "analysis", "update"
            ]
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        logger.info("Sentiment analyzer initialized")
    
    async def analyze_symbol(self, symbol: str, sources: List[SentimentSource] = None) -> Dict[str, Any]:
        """Analyze sentiment for a specific symbol"""
        if sources is None:
            sources = [SentimentSource.NEWS, SentimentSource.ONCHAIN]
        
        # Check cache first
        cache_key = f"{symbol}:{','.join([s.value for s in sources])}"
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            cache_time = datetime.fromisoformat(cached["timestamp"])
            
            if (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl_hours * 3600:
                logger.debug(f"Using cached sentiment for {symbol}")
                return cached["data"]
        
        # Fetch from all sources
        results = {}
        
        for source in sources:
            try:
                source_data = await self._fetch_from_source(symbol, source)
                if source_data:
                    results[source.value] = source_data
            except Exception as e:
                logger.error(f"Failed to fetch {source.value} sentiment for {symbol}: {e}")
        
        # Aggregate sentiment
        aggregated = self._aggregate_sentiment(results)
        
        # Cache results
        self.sentiment_cache[cache_key] = {
            "timestamp": datetime.now().isoformat(),
            "data": aggregated
        }
        
        # Clean old cache entries
        await self._clean_cache()
        
        logger.info(f"Generated sentiment for {symbol}: {aggregated.get('overall_sentiment', 'neutral')}")
        
        return aggregated
    
    async def _fetch_from_source(self, symbol: str, source: SentimentSource) -> Optional[Dict[str, Any]]:
        """Fetch sentiment from a specific source"""
        if source == SentimentSource.NEWS:
            return await self._fetch_news_sentiment(symbol)
        elif source == SentimentSource.TWITTER:
            return await self._fetch_twitter_sentiment(symbol)
        elif source == SentimentSource.REDDIT:
            return await self._fetch_reddit_sentiment(symbol)
        elif source == SentimentSource.ONCHAIN:
            return await self._fetch_onchain_sentiment(symbol)
        else:
            return None
    
    async def _fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch news sentiment"""
        if not self.config.news_api_key:
            logger.warning("News API key not configured")
            return self._get_empty_sentiment("news")
        
        try:
            # Extract coin name from symbol (e.g., SOL from SOL/USDT)
            coin_name = symbol.split("/")[0].lower()
            
            # Use CryptoPanic API or similar
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.config.news_api_key}&currencies={coin_name}"
            
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._analyze_news_data(data, coin_name)
                else:
                    logger.error(f"News API error: {response.status}")
                    return self._get_empty_sentiment("news")
        
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return self._get_empty_sentiment("news")
    
    def _analyze_news_data(self, data: Dict[str, Any], coin_name: str) -> Dict[str, Any]:
        """Analyze news data for sentiment"""
        if "results" not in data:
            return self._get_empty_sentiment("news")
        
        articles = data["results"]
        sentiments = []
        volumes = []
        keywords_found = []
        
        for article in articles[:10]:  # Analyze top 10 articles
            title = article.get("title", "")
            description = article.get("description", "")
            votes = article.get("votes", {})
            
            # Combine title and description
            text = f"{title} {description}".lower()
            
            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Keyword analysis
            keyword_scores = self._analyze_keywords(text)
            
            # Article importance (based on votes)
            importance = (
                votes.get("positive", 0) +
                votes.get("negative", 0) +
                votes.get("important", 0)
            ) / 10  # Normalize
            
            # Overall sentiment for this article
            article_sentiment = self._combine_sentiment_scores(
                polarity, keyword_scores, importance
            )
            
            sentiments.append(article_sentiment)
            volumes.append(importance)
            
            # Track keywords
            for keyword in self._extract_keywords(text):
                if keyword not in keywords_found:
                    keywords_found.append(keyword)
        
        # Aggregate across articles
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            sentiment_std = np.std(sentiments)
            total_volume = sum(volumes)
            
            return {
                "source": "news",
                "sentiment_score": float(avg_sentiment),
                "sentiment_confidence": float(1 - min(sentiment_std, 1)),  # Lower std = higher confidence
                "volume": float(total_volume),
                "keywords": keywords_found[:10],
                "articles_analyzed": len(articles),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return self._get_empty_sentiment("news")
    
    async def _fetch_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch Twitter sentiment"""
        if not self.config.twitter_bearer_token:
            logger.warning("Twitter API token not configured")
            return self._get_empty_sentiment("twitter")
        
        try:
            coin_name = symbol.split("/")[0]
            
            # Twitter API v2 search
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {self.config.twitter_bearer_token}"
            }
            params = {
                "query": f"#{coin_name} crypto -is:retweet",
                "max_results": 50,
                "tweet.fields": "public_metrics,created_at"
            }
            
            async with self.http_session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._analyze_twitter_data(data, coin_name)
                else:
                    logger.error(f"Twitter API error: {response.status}")
                    return self._get_empty_sentiment("twitter")
        
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return self._get_empty_sentiment("twitter")
    
    def _analyze_twitter_data(self, data: Dict[str, Any], coin_name: str) -> Dict[str, Any]:
        """Analyze Twitter data for sentiment"""
        if "data" not in data:
            return self._get_empty_sentiment("twitter")
        
        tweets = data["data"]
        sentiments = []
        engagements = []
        keywords_found = []
        
        for tweet in tweets:
            text = tweet.get("text", "").lower()
            metrics = tweet.get("public_metrics", {})
            
            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Keyword analysis
            keyword_scores = self._analyze_keywords(text)
            
            # Engagement metrics
            engagement = (
                metrics.get("like_count", 0) +
                metrics.get("retweet_count", 0) +
                metrics.get("reply_count", 0)
            ) / 100  # Normalize
            
            # Tweet sentiment
            tweet_sentiment = self._combine_sentiment_scores(
                polarity, keyword_scores, min(engagement, 1)
            )
            
            sentiments.append(tweet_sentiment)
            engagements.append(engagement)
            
            # Track keywords
            for keyword in self._extract_keywords(text):
                if keyword not in keywords_found:
                    keywords_found.append(keyword)
        
        # Aggregate across tweets
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            sentiment_std = np.std(sentiments)
            total_engagement = sum(engagements)
            
            return {
                "source": "twitter",
                "sentiment_score": float(avg_sentiment),
                "sentiment_confidence": float(1 - min(sentiment_std, 1)),
                "engagement": float(total_engagement),
                "keywords": keywords_found[:10],
                "tweets_analyzed": len(tweets),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return self._get_empty_sentiment("twitter")
    
    async def _fetch_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch Reddit sentiment"""
        if not self.config.reddit_client_id:
            logger.warning("Reddit API not configured")
            return self._get_empty_sentiment("reddit")
        
        try:
            coin_name = symbol.split("/")[0].lower()
            
            # Reddit API requires OAuth2
            # For simplicity, return empty for now
            # In production, implement proper Reddit API integration
            
            return self._get_empty_sentiment("reddit")
            
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {e}")
            return self._get_empty_sentiment("reddit")
    
    async def _fetch_onchain_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch on-chain sentiment metrics"""
        try:
            coin_name = symbol.split("/")[0].lower()
            
            # Use CoinGecko API for on-chain metrics
            url = f"https://api.coingecko.com/api/v3/coins/{coin_name}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true",
                "sparkline": "false"
            }
            
            async with self.http_session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._analyze_onchain_data(data, coin_name)
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return self._get_empty_sentiment("onchain")
        
        except Exception as e:
            logger.error(f"Error fetching on-chain sentiment: {e}")
            return self._get_empty_sentiment("onchain")
    
    def _analyze_onchain_data(self, data: Dict[str, Any], coin_name: str) -> Dict[str, Any]:
        """Analyze on-chain data for sentiment"""
        sentiment_score = 0.0
        metrics = {}
        
        try:
            # Market cap rank (lower is better)
            market_cap_rank = data.get("market_cap_rank", 999)
            if market_cap_rank < 10:
                sentiment_score += 0.3
            elif market_cap_rank < 50:
                sentiment_score += 0.1
            
            # Price change percentages
            market_data = data.get("market_data", {})
            
            price_change_24h = market_data.get("price_change_percentage_24h", 0)
            if price_change_24h > 5:
                sentiment_score += 0.2
            elif price_change_24h < -5:
                sentiment_score -= 0.2
            
            price_change_7d = market_data.get("price_change_percentage_7d", 0)
            if price_change_7d > 10:
                sentiment_score += 0.1
            elif price_change_7d < -10:
                sentiment_score -= 0.1
            
            # Volume
            total_volume = market_data.get("total_volume", {}).get("usd", 0)
            market_cap = market_data.get("market_cap", {}).get("usd", 1)
            volume_market_cap_ratio = total_volume / market_cap if market_cap > 0 else 0
            
            if volume_market_cap_ratio > 0.1:  # High volume relative to market cap
                sentiment_score += 0.15
            
            # Community data
            community_data = data.get("community_data", {})
            
            twitter_followers = community_data.get("twitter_followers", 0)
            if twitter_followers > 100000:
                sentiment_score += 0.05
            
            reddit_subscribers = community_data.get("reddit_subscribers", 0)
            if reddit_subscribers > 100000:
                sentiment_score += 0.05
            
            # Developer data
            developer_data = data.get("developer_data", {})
            
            github_stars = developer_data.get("stars", 0)
            if github_stars > 1000:
                sentiment_score += 0.05
            
            forks = developer_data.get("forks", 0)
            if forks > 500:
                sentiment_score += 0.05
            
            # Normalize sentiment score to -1 to 1
            sentiment_score = max(-1, min(1, sentiment_score))
            
            metrics = {
                "market_cap_rank": market_cap_rank,
                "price_change_24h": price_change_24h,
                "price_change_7d": price_change_7d,
                "volume_market_cap_ratio": volume_market_cap_ratio,
                "twitter_followers": twitter_followers,
                "reddit_subscribers": reddit_subscribers,
                "github_stars": github_stars,
                "forks": forks
            }
            
        except Exception as e:
            logger.error(f"Error analyzing on-chain data: {e}")
        
        return {
            "source": "onchain",
            "sentiment_score": float(sentiment_score),
            "sentiment_confidence": 0.7,  # On-chain data is objective
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_keywords(self, text: str) -> Dict[str, float]:
        """Analyze keywords in text"""
        scores = {"positive": 0, "negative": 0, "neutral": 0}
        text_lower = text.lower()
        
        for sentiment_type, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[sentiment_type] += 1
        
        total_keywords = sum(scores.values())
        if total_keywords > 0:
            # Convert to sentiment score: positive increases, negative decreases
            keyword_sentiment = (scores["positive"] - scores["negative"]) / total_keywords
            return {"score": keyword_sentiment, "count": total_keywords}
        else:
            return {"score": 0, "count": 0}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        found_keywords = []
        text_lower = text.lower()
        
        for sentiment_type, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text_lower and keyword not in found_keywords:
                    found_keywords.append(keyword)
        
        return found_keywords[:5]  # Return top 5 keywords
    
    def _combine_sentiment_scores(self, textblob_score: float, 
                                 keyword_analysis: Dict[str, float],
                                 importance: float) -> float:
        """Combine multiple sentiment scores"""
        keyword_score = keyword_analysis.get("score", 0)
        keyword_count = keyword_analysis.get("count", 0)
        
        # Weighted combination
        if keyword_count >= 3:  # If we found enough keywords
            # Trust keywords more when they're abundant
            combined = (keyword_score * 0.7) + (textblob_score * 0.3)
        else:
            # Trust TextBlob more when keywords are scarce
            combined = (keyword_score * 0.3) + (textblob_score * 0.7)
        
        # Adjust by importance
        adjusted = combined * (0.5 + 0.5 * importance)  # Importance boosts confidence
        
        return max(-1, min(1, adjusted))
    
    def _aggregate_sentiment(self, source_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple sources"""
        if not source_results:
            return self._get_empty_sentiment("aggregated")
        
        sentiments = []
        confidences = []
        source_details = {}
        
        for source, data in source_results.items():
            sentiment = data.get("sentiment_score", 0)
            confidence = data.get("sentiment_confidence", 0.5)
            
            # Only include if confidence meets threshold
            if confidence >= self.config.min_confidence:
                sentiments.append(sentiment)
                confidences.append(confidence)
            
            # Store source details
            source_details[source] = {
                "sentiment": sentiment,
                "confidence": confidence,
                "volume": data.get("volume", data.get("engagement", 0))
            }
        
        if sentiments:
            # Weighted average by confidence
            total_confidence = sum(confidences)
            if total_confidence > 0:
                weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_confidence
                avg_confidence = np.mean(confidences)
            else:
                weighted_sentiment = 0
                avg_confidence = 0
        else:
            weighted_sentiment = 0
            avg_confidence = 0
        
        # Determine sentiment category
        if weighted_sentiment > 0.2:
            sentiment_category = "bullish"
        elif weighted_sentiment < -0.2:
            sentiment_category = "bearish"
        else:
            sentiment_category = "neutral"
        
        # Calculate sentiment strength
        sentiment_strength = abs(weighted_sentiment) * avg_confidence
        
        return {
            "overall_sentiment": sentiment_category,
            "sentiment_score": float(weighted_sentiment),
            "sentiment_confidence": float(avg_confidence),
            "sentiment_strength": float(sentiment_strength),
            "source_count": len(source_results),
            "sources": source_details,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_empty_sentiment(self, source: str) -> Dict[str, Any]:
        """Get empty sentiment result for a source"""
        return {
            "source": source,
            "sentiment_score": 0.0,
            "sentiment_confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _clean_cache(self):
        """Clean old entries from cache"""
        now = datetime.now()
        keys_to_delete = []
        
        for key, value in self.sentiment_cache.items():
            cache_time = datetime.fromisoformat(value["timestamp"])
            if (now - cache_time).total_seconds() > self.config.cache_ttl_hours * 3600:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.sentiment_cache[key]
        
        if keys_to_delete:
            logger.debug(f"Cleaned {len(keys_to_delete)} old sentiment cache entries")
    
    async def get_symbol_sentiment_history(self, symbol: str, 
                                          hours: int = 24) -> List[Dict[str, Any]]:
        """Get sentiment history for a symbol"""
        history = []
        now = datetime.now()
        
        for cache_key, cache_data in self.sentiment_cache.items():
            if symbol in cache_key:
                cache_time = datetime.fromisoformat(cache_data["timestamp"])
                
                if (now - cache_time).total_seconds() <= hours * 3600:
                    history.append({
                        "timestamp": cache_data["timestamp"],
                        "data": cache_data["data"]
                    })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        return history
    
    async def close(self):
        """Close HTTP session"""
        if self.http_session:
            await self.http_session.close()
            logger.info("Sentiment analyzer closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            "cache_size": len(self.sentiment_cache),
            "keywords_positive": len(self.crypto_keywords["positive"]),
            "keywords_negative": len(self.crypto_keywords["negative"]),
            "keywords_neutral": len(self.crypto_keywords["neutral"])
        }
