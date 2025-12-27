import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
import time
from urllib.parse import quote
from .config import get_newsapi_key, ConfigurationError
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*possibly delisted.*')


def get_company_name_from_ticker(ticker: str) -> Optional[str]:
    """Get the full company name from ticker symbol using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        # Suppress output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = stock.info
        
        company_name = (
            info.get('longName') or 
            info.get('shortName') or 
            info.get('name')
        )
        
        return company_name
    except Exception:
        return None


def extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain.split('.')[0].title()
    except:
        return 'Unknown'
    

def fetch_web_search_results(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch general web search results from DuckDuckGo (NOT news, just web results).
    This gives us URLs and snippets that we can then use to fetch full articles.
    """
    try:
        from duckduckgo_search import DDGS
        
        print(f"🔍 DuckDuckGo Web Search for: {query}")
        
        ddgs = DDGS()
        # Use .text() for general web search, not .news()
        results = ddgs.text(query, max_results=max_results)
        
        search_results = []
        for result in results:
            search_results.append({
                'title': result.get('title', ''),
                'snippet': result.get('body', '')[:300],
                'url': result.get('href', ''),
                'source': extract_domain(result.get('href', '')),
            })
        
        if search_results:
            print(f"✓ DuckDuckGo Web Search: Found {len(search_results)} results")
        else:
            print(f"⚠ DuckDuckGo Web Search: No results found for '{query}'")
        return search_results
        
    except ImportError:
        print("⚠ DuckDuckGo: Library not installed (pip install duckduckgo-search)")
        return []
    except Exception as e:
        print(f"⚠ DuckDuckGo Web Search error: {type(e).__name__}: {str(e)[:100]}")
        return []

def fetch_from_duckduckgo(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Fetch news from DuckDuckGo search."""
    try:
        from duckduckgo_search import DDGS
        
        print(f"🔍 Searching DuckDuckGo for: {query}")
        
        ddgs = DDGS()
        results = ddgs.news(query, max_results=max_results)
        
        news_items = []
        for result in results:
            news_items.append({
                'title': result.get('title', ''),
                'description': result.get('body', '')[:500],
                'url': result.get('url', ''),
                'published_at': result.get('date', datetime.now().isoformat()),
                'source': result.get('source', extract_domain(result.get('url', ''))),
                'content_snippet': result.get('body', ''),
                'full_content': result.get('body', ''),
                'score': 10,  # Default score for DDG results
            })
        
        if news_items:
            print(f"✓ DuckDuckGo: Found {len(news_items)} articles")
        else:
            print(f"⚠ DuckDuckGo: No articles found for '{query}'")
        return news_items
        
    except ImportError:
        print("⚠ DuckDuckGo: Library not installed (pip install duckduckgo-search)")
        return []
    except Exception as e:
        print(f"⚠ DuckDuckGo error: {type(e).__name__}: {str(e)[:100]}")
        return []


def fetch_from_tavily_news(query: str, days: int = 7, max_results: int = 10) -> List[Dict[str, Any]]:
    """Fetch news using Tavily AI Search API with timeout protection."""
    try:
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        if not tavily_api_key:
            print("Tavily: API key not configured")
            return []
        
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=tavily_api_key)
        
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_domains=[
                "finance.yahoo.com", "reuters.com", "bloomberg.com", 
                "cnbc.com", "marketwatch.com", "seekingalpha.com",
                "fool.com", "benzinga.com", "investing.com", "forbes.com"
            ],
            topic="news"
        )
        
        news_items = []
        for result in response.get('results', []):
            news_items.append({
                'title': result.get('title', ''),
                'description': result.get('content', '')[:500],
                'url': result.get('url', ''),
                'published_at': result.get('published_date', datetime.now().isoformat()),
                'source': extract_domain(result.get('url', '')),
                'content_snippet': result.get('content', ''),
                'full_content': result.get('content', ''),
                'score': result.get('score', 0),
            })
        
        if news_items:
            print(f"✓ Tavily: Found {len(news_items)} articles")
        else:
            print(f"⚠ Tavily: No articles found for '{query}'")
        return news_items
        
    except ImportError:
        print("Tavily: Library not installed")
        return []
    except Exception as e:
        print(f"⚠ Tavily error: {type(e).__name__}: {str(e)[:100]}")
        return []


def fetch_from_newsapi(query: str, days: int = 7) -> List[Dict[str, Any]]:
    """Fetch news from NewsAPI."""
    try:
        api_key = get_newsapi_key()
        newsapi = NewsApiClient(api_key=api_key)

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            page_size=20
        )

        news_items = []
        if articles.get('status') == 'ok':
            for article in articles['articles']:
                if article.get('title') and article['title'] != '[Removed]':
                    news_items.append({
                        'title': article['title'],
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                        'content_snippet': article.get('content', '')
                    })
        
        if news_items:
            print(f"✓ NewsAPI: Found {len(news_items)} articles")
        else:
            print(f"⚠ NewsAPI: No articles found for '{query}'")
        return news_items

    except Exception as e:
        print(f"⚠ NewsAPI error: {type(e).__name__}: {str(e)[:100]}")
        return []


def fetch_from_google_news(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """Fetch news from Google News RSS."""
    try:
        import feedparser
        
        encoded_query = quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(rss_url)
        
        news_items = []
        for entry in feed.entries[:num_results]:
            news_items.append({
                'title': entry.get('title', ''),
                'description': entry.get('summary', ''),
                'url': entry.get('link', ''),
                'published_at': entry.get('published', ''),
                'source': 'Google News',
                'content_snippet': entry.get('summary', '')
            })
        
        if news_items:
            print(f"✓ Google News: Found {len(news_items)} articles")
        else:
            print(f"⚠ Google News: No articles found for '{query}'")
        return news_items
        
    except Exception as e:
        print(f"⚠ Google News error: {type(e).__name__}: {str(e)[:100]}")
        return []


def fetch_from_yahoo_finance(ticker: str) -> List[Dict[str, Any]]:
    """Fetch news from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            news = stock.news
        
        news_items = []
        for item in news[:10]:
            news_items.append({
                'title': item.get('title', ''),
                'description': item.get('summary', ''),
                'url': item.get('link', ''),
                'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                'source': item.get('publisher', 'Yahoo Finance'),
                'content_snippet': item.get('summary', '')
            })
        
        if news_items:
            print(f"✓ Yahoo Finance: Found {len(news_items)} articles")
        else:
            print(f"⚠ Yahoo Finance: No articles found for {ticker}")
        return news_items
        
    except Exception as e:
        print(f"⚠ Yahoo Finance error: {type(e).__name__}: {str(e)[:100]}")
        return []


def get_price_history(ticker: str, period: str = "1mo") -> Optional[object]:
    """Fetch historical price data."""
    try:
        stock = yf.Ticker(ticker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history = stock.history(period=period)

        if history.empty:
            return None

        return history

    except Exception:
        return None


def get_stock_currency(ticker: str) -> str:
    """Get currency with fallback."""
    try:
        stock = yf.Ticker(ticker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = stock.info
        currency = info.get('currency', 'USD')
        return currency
    except Exception:
        return 'USD'


def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Get comprehensive stock info."""
    try:
        stock = yf.Ticker(ticker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = stock.info
        
        return {
            'currency': info.get('currency', 'USD'),
            'market_cap': info.get('marketCap'),
            'exchange': info.get('exchange'),
            'long_name': info.get('longName', ticker),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'country': info.get('country')
        }
    except Exception:
        return {
            'currency': 'USD',
            'market_cap': None,
            'exchange': None,
            'long_name': ticker,
            'sector': None,
            'industry': None,
            'country': None
        }


def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate articles."""
    seen_titles = set()
    unique_articles = []
    
    for article in articles:
        title = article.get('title', '').lower().strip()
        title_key = ''.join(title.split())[:100]
        
        if title_key not in seen_titles and title:
            seen_titles.add(title_key)
            unique_articles.append(article)
    
    return unique_articles


def get_comprehensive_news(
    query: str, 
    days: int = 30,
    fetch_full_content: bool = False,
    max_articles: int = 10,
    use_tavily: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch comprehensive news from multiple sources.
    IMPORTANT: This function should only be called ONCE per analysis.
    Restricted to 10 articles maximum for performance.
    """
    print(f"\n{'='*60}")
    print(f"Fetching news for {query} (max 10 articles)")
    print(f"{'='*60}")
    
    # Get company name

    
    # CHANGED: Formulate proper query for stock details
    print(f"Search Query: {query}")
    
    all_articles = []
    
    # 1. Tavily - Priority (if available)
    if use_tavily:
        print("\n[1/4] Trying Tavily AI Search...")
        tavily_articles = fetch_from_tavily_news(query, days, max_results=10)
        if tavily_articles:
            all_articles.extend(tavily_articles)
            print(f"✓ Tavily successful: {len(tavily_articles)} articles")
            # Early exit if we have enough articles
            if len(all_articles) >= max_articles:
                unique_articles = deduplicate_articles(all_articles)[:max_articles]
                print(f"\n{'='*60}")
                print(f"articles are {unique_articles}")
                print(f"✓ Final: {len(unique_articles)} articles (early exit - quota reached)")
                print(f"{'='*60}\n")
                return unique_articles
    
    # 2. Yahoo Finance
    print("\n[2/4] Trying Yahoo Finance...")
    yahoo_articles = fetch_from_yahoo_finance(query)
    if yahoo_articles:
        all_articles.extend(yahoo_articles)
        if len(all_articles) >= max_articles:
            unique_articles = deduplicate_articles(all_articles)[:max_articles]
            print(f"\n{'='*60}")
            print(f"articles are {unique_articles}")
            print(f"✓ Final: {len(unique_articles)} articles (early exit - quota reached)")
            print(f"{'='*60}\n")
            return unique_articles
    
    # 3. NewsAPI
    print("\n[3/4] Trying NewsAPI...")
    newsapi_articles = fetch_from_newsapi(query, days)
    if newsapi_articles:
        all_articles.extend(newsapi_articles)
        print(f"✓ NewsAPI successful: {len(newsapi_articles)} articles")
        if len(all_articles) >= max_articles:
            unique_articles = deduplicate_articles(all_articles)[:max_articles]
            print(f"\n{'='*60}")
            print(f"articles are {unique_articles}")
            print(f"✓ Final: {len(unique_articles)} articles (early exit - quota reached)")
            print(f"{'='*60}\n")
            return unique_articles
    
    # 4. Google News (optional) - only if we need more articles
    if len(all_articles) < max_articles:
        print("\n[4/4] Trying Google News...")
        try:
            import feedparser
            google_articles = fetch_from_google_news(query)
            if google_articles:
                all_articles.extend(google_articles)
        except ImportError:
            pass
    
    # Deduplicate
    unique_articles = deduplicate_articles(all_articles)
    print(f"articles are {unique_articles}")
    # Sort by score/date
    unique_articles.sort(
        key=lambda x: (x.get('score', 0), x.get('published_at', '')), 
        reverse=True
    )
    
    # Limit to max_articles
    unique_articles = unique_articles[:max_articles]
    
    print(f"\n{'='*60}")
    print(f"✓ Final: {len(unique_articles)} unique articles")
    print(f"{'='*60}\n")
    
    return unique_articles


def get_news_for_llm(ticker: str, days: int = 30, max_articles: int = 10) -> str:
    """
    Get formatted news for LLM.
    NOTE: This calls get_comprehensive_news internally, so don't call both!
    """
    articles = get_comprehensive_news(
        ticker, 
        days, 
        fetch_full_content=False,
        max_articles=max_articles,
        use_tavily=True
    )
    
    if not articles:
        return f"No recent news found for {ticker} in the last {days} days."
    
    formatted_news = f"News Analysis for {ticker}\n{'='*80}\n\n"
    formatted_news += f"Found {len(articles)} relevant articles:\n\n"
    
    for i, article in enumerate(articles, 1):
        formatted_news += f"Article {i}:\n"
        formatted_news += f"Title: {article.get('title', 'N/A')}\n"
        formatted_news += f"Source: {article.get('source', 'N/A')}\n"
        formatted_news += f"Date: {article.get('published_at', 'N/A')[:10]}\n"
        
        if article.get('score'):
            formatted_news += f"Relevance: {article['score']:.2f}\n"
        
        content = article.get('full_content') or article.get('description') or article.get('content_snippet', '')
        if content:
            formatted_news += f"Content: {content[:600]}...\n"
        
        formatted_news += "\n" + "-"*80 + "\n\n"
    
    return formatted_news


def get_news(ticker: str, days: int = 30) -> List[str]:
    """Backward compatible - returns headlines only."""
    articles = get_comprehensive_news(ticker, days, fetch_full_content=False, max_articles=20)
    return [article['title'] for article in articles if article.get('title')]


def detect_query_type(query: str) -> Dict[str, Any]:
    """
    Detect the type of query and extract relevant information.
    Returns: {type, is_ticker, ticker, company_name, category}
    """
    query_upper = query.upper().strip()
    
    # Check if it's a ticker (typically 1-5 uppercase letters, optional exchange suffix)
    import re
    ticker_pattern = r'^[A-Z]{1,5}(?:\.[A-Z]{2})?$'
    is_ticker = bool(re.match(ticker_pattern, query_upper))
    
    query_type = "ticker" if is_ticker else "general"
    
    return {
        "type": query_type,
        "is_ticker": is_ticker,
        "ticker": query_upper if is_ticker else None,
        "company_name": None,
        "original_query": query,
        "category": "finance"
    }


def format_research_for_llm(research_data: Dict[str, Any]) -> str:
    """
    Format research data into a comprehensive text document for LLM analysis.
    """
    articles = research_data.get('articles', [])
    query = research_data.get('query', 'Unknown')
    web_results = research_data.get('web_results', [])
    query_analysis = research_data.get('query_analysis', {})
    summary_info = research_data.get('summary_info', {})
    search_queries = research_data.get('search_queries_used', [])
    
    formatted = f"""
FINANCIAL RESEARCH DATA

QUERY INFORMATION

Original Query: {query}
Query Type: {query_analysis.get('type', 'general')}
Search Queries Used: {', '.join(search_queries) if search_queries else 'Original query'}
Analysis Date: {datetime.now().strftime('%B %d, %Y')}

DATA SUMMARY
Web results: {web_results}
Total Articles: {len(articles)}
Average Relevance Score: {summary_info.get('avg_relevance', 0):.2f}/20.0
Date Range: {summary_info.get('date_range', 'Last 30 days')}
Primary Sources: {', '.join(summary_info.get('top_sources', []))}

ARTICLE DETAILS

"""
    
    for i, article in enumerate(articles, 1):
        formatted += f"\n[Article {i}]"
        formatted += f"\nTitle: {article.get('title', 'N/A')}"
        formatted += f"\nSource: {article.get('source', 'N/A')}"
        formatted += f"\nPublished: {article.get('published_at', 'N/A')[:10]}"
        
        if article.get('score'):
            formatted += f"\nRelevance: {article.get('score', 0):.2f}"
        
        description = article.get('description') or article.get('content_snippet') or article.get('full_content', '')
        if description:
            formatted += f"\nSummary: {description[:400]}"
        
        formatted += "\n"
    
    formatted += f"""
{'='*80}
END OF RESEARCH DATA
{'='*80}
"""
    
    return formatted


def get_intelligent_research(
    query: str,
    days: int = 30,
    max_articles: int = 10,
    fetch_full_content: bool = False
) -> Dict[str, Any]:
    """
    Intelligent research function that works with ANY query type.
    Automatically detects whether query is a ticker, company name, or general topic.
    
    IMPROVED LOGIC:
    - For ticker queries: Formulates proper search query like "what are the stock details of {company_name}"
    - For general queries: Uses DuckDuckGo first, then supplements with other sources
    
    Args:
        query: The search query (ticker, company name, or topic)
        days: Number of days of news to fetch
        max_articles: Maximum number of articles to return
        fetch_full_content: Whether to fetch full article content
    
    Returns:
        Dictionary containing:
        - articles: List of article dictionaries
        - query_analysis: Analysis of the query type
        - summary_info: Summary statistics about the articles
        - search_queries_used: List of search queries that were used
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 Intelligent Research for: {query}")
    print(f"{'='*60}\n")
    
    # Step 1: Detect query type
    query_analysis = detect_query_type(query)
    print(f"Query Type: {query_analysis.get('type')}")
    
    is_ticker = query_analysis.get('is_ticker', False)
    all_articles = []
    search_queries_used = []
    
    # Step 2: Handle based on query type
    if is_ticker:
        # TICKER QUERY: Formulate proper search query
        ticker = query_analysis['ticker']
        company_name = get_company_name_from_ticker(ticker)
        
        if company_name:
            query_analysis['company_name'] = company_name
            search_query = f"what are the stock details of {company_name} stock price site:reuters.com OR site:bloomberg.com OR site:finance.yahoo.com"
            print(f"Company: {company_name}")
        else:
            search_query = f"what are the stock details of {ticker} stock price site:reuters.com OR site:bloomberg.com OR site:finance.yahoo.com"
        
        print(f"Formulated Query: {search_query}")
        search_queries_used.append(search_query)
        
        # Use comprehensive news fetching for tickers
        articles = get_comprehensive_news(
            query,
            days=days,
            fetch_full_content=fetch_full_content,
            max_articles=max_articles,
            use_tavily=True
        )
        all_articles.extend(articles)
        
    else:
        # GENERAL QUERY: Use DuckDuckGo first
        print(f"📰 General query detected - Using DuckDuckGo first")
        search_queries_used.append(query)
        
        web_results = fetch_web_search_results(query, max_results=10)
        print("web_results are", web_results)
        print(f"\n[1/4] Trying DuckDuckGo...")
        refined_query = f"{query} stock price site:reuters.com OR site:bloomberg.com OR site:finance.yahoo.com"
        ddg_articles = fetch_from_duckduckgo(refined_query, max_results=max_articles)
        
        if web_results:
            print(f"✓ Web Search: Found {len(web_results)} results")
            print(f"   These will be shown to LLM for context")
        
        # STEP 2: Now fetch actual NEWS ARTICLES from multiple sources
        print(f"\n[Step 2/2] Fetching News Articles...")

        if ddg_articles:
            all_articles.extend(ddg_articles)
            print(f"✓ DuckDuckGo: {len(ddg_articles)} articles")
            
            # Early exit if we have enough high-quality articles
            if len(all_articles) >= max_articles:
                print(f"✓ Sufficient articles found from DuckDuckGo")
        
        # 2. If not enough articles, try Tavily
        if len(all_articles) < max_articles:
            print(f"\n[2/4] Trying Tavily AI Search...")
            tavily_articles = fetch_from_tavily_news(refined_query, days=days, max_results=max_articles)
            if tavily_articles:
                all_articles.extend(tavily_articles)
                print(f"✓ Tavily: {len(tavily_articles)} articles")
        
        # 3. If still not enough, try NewsAPI
        if len(all_articles) < max_articles:
            print(f"\n[3/4] Trying NewsAPI...")
            newsapi_articles = fetch_from_newsapi(refined_query, days=days)
            if newsapi_articles:
                all_articles.extend(newsapi_articles)
                print(f"✓ NewsAPI: {len(newsapi_articles)} articles")
        
        # 4. If still not enough, try Google News
        if len(all_articles) < max_articles:
            print(f"\n[4/4] Trying Google News...")
            try:
                google_articles = fetch_from_google_news(refined_query, num_results=max_articles)
                if google_articles:
                    all_articles.extend(google_articles)
                    print(f"✓ Google News: {len(google_articles)} articles")
            except:
                pass
    
    # Step 3: Deduplicate and limit
    unique_articles = deduplicate_articles(all_articles)
    
    # Sort by relevance/date
    unique_articles.sort(
        key=lambda x: (x.get('score', 0), x.get('published_at', '')),
        reverse=True
    )
    
    # Limit to max_articles
    unique_articles = unique_articles[:max_articles]
    
    print(f"\n✓ Total unique articles: {len(unique_articles)}")
    if web_results:
        print(f"✓ Web search results: {len(web_results)}")
    # Step 4: Calculate summary info
    sources = [article.get('source', 'Unknown') for article in unique_articles]
    unique_sources = list(set(sources))
    
    relevance_scores = [article.get('score', 5) for article in unique_articles if article.get('score')]
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    
    # Get date range
    dates = [article.get('published_at', '') for article in unique_articles if article.get('published_at')]
    date_range = f"{min(dates)[:10]} to {max(dates)[:10]}" if dates else "Last 30 days"
    
    summary_info = {
        "total_articles": len(unique_articles),
        "total_web_results": len(web_results),
        "unique_sources": len(unique_sources),
        "top_sources": unique_sources[:5],
        "avg_relevance": avg_relevance,
        "date_range": date_range
    }
    
    # Step 5: Return structured result (UNCHANGED - keeping same structure for LLM compatibility)
    result = {
        "query": query,
        "query_analysis": query_analysis,
        "web_results": web_results,
        "articles": unique_articles,
        "summary_info": summary_info,
        "search_queries_used": search_queries_used,
        "formatted_content": ""  # Will be filled by format_research_for_llm
    }
    
    print(f"\n{'='*60}")
    print(f"✓ Research Complete")
    print(f"{'='*60}\n")
    
    return result