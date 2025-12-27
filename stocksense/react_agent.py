from typing import Dict, List, Optional, TypedDict, Literal, Any
from datetime import datetime
import traceback
import logging
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from .config import get_chat_llm
from .data_collectors import (  # Updated import
    get_intelligent_research,
    format_research_for_llm,
    get_price_history,
    get_stock_info
)
from .analyzer import analyze_sentiment_of_headlines
from .database import save_analysis
from .price_tools import PriceDataTools

# Setup logging
logger = logging.getLogger(__name__)
logger.info("🚀 React Agent module loading...")


class AgentState(TypedDict):
    """Enhanced state for universal research agent."""
    messages: List[BaseMessage]
    query: str  # Changed from ticker
    query_type: str  # auto, ticker, news, company
    research_data: Dict[str, Any]  # Complete research results
    articles: List[Dict[str, Any]]
    price_data: List[Dict[str, Any]]
    chart_data: Optional[Dict[str, Any]]  # Added for chart JSON
    technical_analysis: Optional[Dict[str, Any]]  # Added for technical indicators
    current_price: Optional[float]  # Added for current price
    currency: Optional[str]  # Added for currency
    price_analysis: Optional[Dict[str, Any]]  # Added for price analysis
    sentiment_report: str
    summary: str
    reasoning_steps: List[str]
    tools_used: List[str]
    iterations: int
    web_results: List[Dict[str, Any]]
    max_iterations: int
    final_decision: str
    error: Optional[str]
    has_sufficient_data: bool
    should_fetch_price_data: bool


@tool
def fetch_stock_price_data(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Fetch comprehensive stock price data including charts, technical indicators, 
    and analysis. Use this when user asks about stock prices, charts, 
    technical analysis, or wants to see price history.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        Dictionary with price data, charts, and technical analysis
    """
    return PriceDataTools.fetch_price_data(ticker, period)


@tool
def get_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Get fundamental data for a stock: financials, ratios, company info.
    Use this when user asks about company fundamentals, financials, 
    or valuation metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "tool": "get_stock_fundamentals",
            "status": "success",
            "ticker": ticker,
            "fundamentals": {
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "pb_ratio": info.get('priceToBook'),
                "dividend_yield": info.get('dividendYield'),
                "beta": info.get('beta'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "employees": info.get('fullTimeEmployees'),
                "website": info.get('website'),
                "summary": info.get('longBusinessSummary', '')[:500]
            }
        }
    except Exception as e:
        return {
            "tool": "get_stock_fundamentals",
            "status": "error",
            "message": str(e)
        }
    

@tool
def intelligent_research_tool(query: str, days: int = 30, max_articles: int = 10) -> Dict:
    """
    Universal financial research tool - works for ANY query.
    Can handle: tickers, company names, market trends, news queries, etc.
    For TICKER queries: Formulates proper search query and fetches news articles
    For GENERAL queries: 
        1. First does DuckDuckGo web search to get context
        2. Then fetches actual news articles from multiple source

    Returns both web_results and articles for comprehensive analysis.
    """
    try:
        print(f"\n🔍 Starting intelligent research for: {query}")
        
        research_data = get_intelligent_research(
            query=query,
            days=days,
            max_articles=max_articles,
            fetch_full_content=False
        )
        
        if research_data and research_data.get('articles'):
            articles = research_data['articles']
            query_analysis = research_data['query_analysis']
            web_results = research_data.get('web_results', [])
            summary_info = research_data['summary_info']
            
            # Format for LLM
            formatted_content = format_research_for_llm(research_data)
            
            return {
                "success": True,
                "query": query,
                "query_analysis": query_analysis,
                "articles": articles,
                "web_results": web_results,
                "formatted_content": formatted_content,
                "summary_info": summary_info,
                "search_queries_used": research_data.get('search_queries_used', []),
                "count": len(articles),
                "message": f"✓ Found {len(articles)} relevant articles"
            }
        else:
            return {
                "success": False,
                "error": f"No relevant information found for: {query}",
                "articles": [],
                "web_results": [],
                "formatted_content": f"Unable to find relevant information for: {query}",
                "message": "⚠ No results found"
            }
            
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"❌ Research error: {type(e).__name__}: {error_msg}")
        return {
            "success": False,
            "error": f"{type(e).__name__}: {error_msg}",
            "articles": [],
            "web_results": [],
            "formatted_content": f"Research failed: {type(e).__name__}",
            "message": f"✗ Research failed"
        }


@tool
def fetch_price_data_tool(ticker: str, period: str = "1mo") -> Dict:
    """Fetch historical price data (only for ticker queries)."""
    try:
        ticker = ticker.upper().strip()
        print(f"\n📈 Fetching price data for {ticker}...")
        
        df = get_price_history(ticker, period=period)

        if df is None or df.empty:
            return {
                "success": False,
                "error": "No price data available",
                "price_data": [],
                "message": "⚠ No price data found"
            }
        
        df_reset = df.reset_index()
        df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')
        
        price_data = []
        for record in df_reset.to_dict(orient='records'):
            price_record = {
                'Date': record['Date'],
                'Open': float(record['Open']) if pd.notna(record['Open']) else None,
                'High': float(record['High']) if pd.notna(record['High']) else None,
                'Low': float(record['Low']) if pd.notna(record['Low']) else None,
                'Close': float(record['Close']) if pd.notna(record['Close']) else None,
                'Volume': int(record['Volume']) if pd.notna(record['Volume']) else None
            }
            price_data.append(price_record)
        
        return {
            "success": True,
            "price_data": price_data,
            "data_points": len(price_data),
            "message": f"✓ Fetched {len(price_data)} price data points"
        }
    except Exception as e:
        print(f"Price fetch error: {type(e).__name__}")
        return {
            "success": False,
            "error": f"{type(e).__name__}",
            "price_data": [],
            "message": "✗ Price data fetch failed"
        }


@tool
def analyze_sentiment_tool(articles_list: List[Dict], query: str) -> Dict:
    """Analyze sentiment from articles."""
    try:
        print(f"\n🤖 Analyzing sentiment for {query}...")
        
        if not articles_list or len(articles_list) == 0:
            return {
                "success": False,
                "error": "No articles provided",
                "sentiment_report": "Unable to perform sentiment analysis",
                "message": "⚠ No articles to analyze"
            }

        # Convert articles to formatted text
        articles_text = "\n\n".join([
            f"Title: {article.get('title', '')}\n"
            f"Source: {article.get('source', '')}\n"
            f"Content: {article.get('description', '') or article.get('content_snippet', '')}"
            for article in articles_list[:10]
        ])
        
        if len(articles_text) < 100:
            return {
                "success": False,
                "error": "Insufficient content",
                "sentiment_report": "Unable to perform sentiment analysis",
                "message": "⚠ Insufficient content"
            }

        sentiment_report = analyze_sentiment_of_headlines([articles_text])
        
        return {
            "success": True,
            "sentiment_report": sentiment_report,
            "message": "✓ Sentiment analysis complete"
        }
    except Exception as e:
        print(f"Sentiment error: {type(e).__name__}")
        
        # Fallback keyword analysis
        try:
            combined_text = " ".join([
                f"{article.get('title', '')} {article.get('description', '')}"
                for article in articles_list
            ]).lower()
            
            positive_words = ['surge', 'soar', 'rally', 'bullish', 'up', 'gain', 'strong', 'growth']
            negative_words = ['crash', 'plunge', 'bearish', 'down', 'decline', 'weak', 'risk']
            
            pos_count = sum(1 for word in positive_words if word in combined_text)
            neg_count = sum(1 for word in negative_words if word in combined_text)
            
            if pos_count > neg_count * 1.5:
                overall = "POSITIVE"
            elif neg_count > pos_count * 1.5:
                overall = "NEGATIVE"
            else:
                overall = "NEUTRAL"
            
            fallback_report = f"""
SENTIMENT ANALYSIS for {query}

Overall Sentiment: {overall}
Positive Indicators: {pos_count}
Negative Indicators: {neg_count}
Articles Analyzed: {len(articles_list)}

Note: Based on keyword analysis
            """.strip()
            
            return {
                "success": True,
                "sentiment_report": fallback_report,
                "message": "✓ Sentiment analysis complete (keyword-based)"
            }
        except:
            return {
                "success": False,
                "error": f"{type(e).__name__}",
                "sentiment_report": "Sentiment analysis unavailable",
                "message": "✗ Sentiment analysis failed"
            }


tools = [
    fetch_stock_price_data,
    get_stock_fundamentals,
    intelligent_research_tool,
    fetch_price_data_tool,
    analyze_sentiment_tool
]


def detect_query_type(query: str) -> Dict[str, Any]:
    """Detect if query is a ticker symbol."""
    import re
    ticker_pattern = r'^[A-Z]{1,5}(?:\.[A-Z]{2})?$'
    is_ticker = bool(re.match(ticker_pattern, query.upper().strip()))
    
    if is_ticker:
        return {"type": "ticker", "ticker": query.upper(), "is_ticker": True}
    return {"type": "general", "is_ticker": False}


def should_fetch_price_data(state: AgentState) -> str:
    """
    Determine if we should fetch price data based on the query.
    """
    query = state["query"].lower()
    
    # Check if query is a ticker
    query_analysis = detect_query_type(state["query"])
    is_ticker = query_analysis.get("is_ticker", False)
    
    # Keywords that indicate price data is needed
    price_keywords = [
        'price', 'chart', 'graph', 'technical', 'candle', 'ohlc',
        'moving average', 'ma', 'rsi', 'volume', 'trend', 'support',
        'resistance', 'high', 'low', 'close', 'open', 'stock price',
        'how much is', 'current price', 'historical', 'history', 'technical analysis'
    ]
    
    # If it's a ticker query OR query mentions price-related terms
    if is_ticker or any(keyword in query for keyword in price_keywords):
        return "fetch_price_data"
    
    return "continue_analysis"


def fetch_price_data_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to fetch price data when needed.
    """
    # Extract ticker from query
    query_analysis = detect_query_type(state["query"])
    ticker = query_analysis.get("ticker") if query_analysis.get("is_ticker") else None
    
    if not ticker:
        # Try to extract ticker from query analysis in research data
        research_data = state.get("research_data", {})
        query_analysis = research_data.get("query_analysis", {})
        if query_analysis.get("is_ticker"):
            ticker = query_analysis.get("ticker")
    
    if not ticker:
        return {
            "price_data": [],
            "chart_data": None,
            "technical_analysis": None,
            "current_price": None,
            "currency": None,
            "price_analysis": None,
            "reasoning_steps": state.get("reasoning_steps", []) + ["No ticker identified for price data"]
        }
    
    print(f"[Agent] Fetching comprehensive price data for {ticker}...")
    
    # Use the comprehensive price data tool
    try:
        result = fetch_stock_price_data.invoke({"ticker": ticker, "period": "6mo"})
        
        if result.get("status") == "success":
            # Update tools used
            current_tools_used = state.get("tools_used", [])
            current_tools_used.append("fetch_stock_price_data")
            
            current_reasoning_steps = state.get("reasoning_steps", [])
            current_reasoning_steps.extend([
                f"Fetched comprehensive price data for {ticker}",
                f"Current price: {result.get('current_price')} {result.get('currency')}",
                f"Price change: {result.get('price_change_pct', 0):.2f}%",
                f"Technical indicators: RSI={result.get('technical_indicators', {}).get('rsi', 'N/A')}"
            ])
            
            return {
                "price_data": result.get("price_data", []),
                "chart_data": result.get("chart_json", {}),
                "technical_analysis": result.get("technical_indicators", {}),
                "current_price": result.get("current_price"),
                "currency": result.get("currency"),
                "price_analysis": result.get("analysis", {}),
                "tools_used": current_tools_used,
                "reasoning_steps": current_reasoning_steps
            }
        else:
            current_reasoning_steps = state.get("reasoning_steps", [])
            current_reasoning_steps.append(
                f"Failed to fetch price data for {ticker}: {result.get('message', 'Unknown error')}"
            )
            
            return {
                "price_data": [],
                "chart_data": None,
                "technical_analysis": None,
                "current_price": None,
                "currency": None,
                "price_analysis": None,
                "reasoning_steps": current_reasoning_steps
            }
    except Exception as e:
        current_reasoning_steps = state.get("reasoning_steps", [])
        current_reasoning_steps.append(f"Price data fetch error: {str(e)[:100]}")
        
        return {
            "price_data": [],
            "chart_data": None,
            "technical_analysis": None,
            "current_price": None,
            "currency": None,
            "price_analysis": None,
            "reasoning_steps": current_reasoning_steps
        }


def generate_price_insight(price_data, tech_analysis, price_analysis):
    """Generate human-readable price insights."""
    if not price_data or not tech_analysis:
        return ""
    
    insights = []
    
    # Current price
    if price_data and len(price_data) > 0:
        latest = price_data[-1]
        close_price = latest.get('close') or latest.get('Close')
        if close_price:
            insights.append(f"• Latest close: ${close_price:.2f}")
    
    # Trend
    if price_analysis and price_analysis.get("trend"):
        trend = price_analysis["trend"]
        trend_map = {
            "strong_uptrend": "Strong uptrend 📈",
            "uptrend": "Uptrend ↗️",
            "downtrend": "Downtrend ↘️",
            "strong_downtrend": "Strong downtrend 📉",
            "consolidation": "Consolidating ↔️"
        }
        insights.append(f"• Trend: {trend_map.get(trend, trend)}")
    
    # RSI
    if tech_analysis and tech_analysis.get("rsi"):
        rsi = tech_analysis["rsi"]
        if rsi > 70:
            insights.append(f"• RSI: {rsi:.1f} (Overbought) ⚠️")
        elif rsi < 30:
            insights.append(f"• RSI: {rsi:.1f} (Oversold) ✅")
        else:
            insights.append(f"• RSI: {rsi:.1f} (Neutral)")
    
    # Support/Resistance
    if tech_analysis and tech_analysis.get("support") and tech_analysis.get("resistance"):
        insights.append(f"• Key levels: Support ${tech_analysis['support']:.2f}, Resistance ${tech_analysis['resistance']:.2f}")
    
    # Moving averages
    if tech_analysis:
        if tech_analysis.get("ma20"):
            insights.append(f"• MA20: ${tech_analysis['ma20']:.2f}")
        if tech_analysis.get("ma50"):
            insights.append(f"• MA50: ${tech_analysis['ma50']:.2f}")
    
    return "\n".join(insights)


def generate_final_analysis(state: AgentState) -> Dict[str, Any]:
    """Generate final analysis including price data if available."""
    
    summary = state.get("summary", "")
    sentiment_report = state.get("sentiment_report", "")
    
    # Build final output
    final_output = {
        "ticker": state.get("research_data", {}).get("query_analysis", {}).get("ticker"),
        "query": state.get("query", ""),
        "summary": summary,
        "sentiment_report": sentiment_report,
        "price_data": state.get("price_data", []),
        "chart_json": state.get("chart_data"),
        "technical_analysis": state.get("technical_analysis"),
        "current_price": state.get("current_price"),
        "currency": state.get("currency"),
        "price_analysis": state.get("price_analysis", {}),
        "web_results": state.get("web_results", []),
        "reasoning_steps": state.get("reasoning_steps", []),
        "tools_used": state.get("tools_used", []),
        "iterations": state.get("iterations", 0),
        "timestamp": datetime.now().isoformat(),
        "research_data": state.get("research_data", {}),
        "articles": state.get("articles", []),
        "query_analysis": state.get("research_data", {}).get("query_analysis", {}),
        "summary_info": state.get("research_data", {}).get("summary_info", {}),
        "search_queries_used": state.get("research_data", {}).get("search_queries_used", []),
        "has_sufficient_data": state.get("has_sufficient_data", False),
        "error": state.get("error")
    }
    
    # Enhance summary with price insights if available
    if state.get("price_data"):
        price_insight = generate_price_insight(
            state.get("price_data"),
            state.get("technical_analysis"),
            state.get("price_analysis", {})
        )
        if price_insight:
            final_output["summary"] += f"\n\n📊 Price Analysis:\n{price_insight}"
    
    return final_output


def create_react_agent() -> StateGraph:
    logger.info("📌 Creating ReAct Agent...")
    try:
        logger.info("🔌 Initializing LLM (gemini-2.5-flash-lite)...")
        llm = get_chat_llm(
            model="gemini-2.5-flash-lite",
            temperature=0.1,
            max_output_tokens=2048
        )
        logger.info("✅ LLM initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM: {type(e).__name__}: {e}")
        logger.error(f"   Full error: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        print(f"Failed to initialize LLM: {type(e).__name__}")
        raise
    
    logger.debug("🔧 Binding tools to LLM...")
    llm_with_tools = llm.bind_tools(tools)
    logger.debug("✅ Tools bound successfully")

    def agent_node(state: AgentState) -> AgentState:
        """Main agent reasoning node."""
        messages = state["messages"]
        query = state["query"]
        query_type = state.get("query_type", "auto")
        iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 5)

        if iterations >= max_iterations:
            has_data = len(state.get('articles', [])) > 0
            
            if has_data:
                return {
                    **state,
                    "final_decision": "COMPLETE",
                    "has_sufficient_data": True
                }
            else:
                return {
                    **state,
                    "final_decision": "MAX_ITERATIONS_REACHED",
                    "error": "Maximum iterations reached",
                    "has_sufficient_data": False
                }

        has_articles = len(state.get('articles', [])) > 0
        has_sentiment = bool(state.get('sentiment_report'))
        research_data = state.get('research_data', {})
        query_analysis = research_data.get('query_analysis', {}) if research_data else {}
        is_ticker_query = query_analysis.get('type') == 'ticker'
        
        # Check if we have price data already
        has_price_data = len(state.get('price_data', [])) > 0
        has_chart_data = bool(state.get('chart_data'))

        reasoning_prompt = f"""
You are a financial research analyst. Analyze the following query comprehensively:

QUERY: {query}
QUERY TYPE: {query_type}
ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d')}

AVAILABLE DATA:
- Articles: {len(state.get('articles', []))} found
- Sentiment: {'Analyzed' if has_sentiment else 'Not analyzed'}
- Price Data: {'Available' if has_price_data else 'Not available'}
- Chart Data: {'Available' if has_chart_data else 'Not available'}
- Query Analysis: {'Available' if query_analysis else 'Not available'}
- Is Ticker Query: {is_ticker_query}
- Tools Used: {', '.join(set(state.get('tools_used', []))) if state.get('tools_used') else 'None'}
- Iteration: {iterations + 1}/{max_iterations}

YOUR TASK - FOLLOW THIS ORDER:

ACTION 1 - IF articles = 0 (DO THIS FIRST):
→ Use intelligent_research_tool with query="{query}"
→ This tool automatically detects query type and searches appropriate sources
→ Wait for results

ACTION 2 - IF articles > 0 AND sentiment not analyzed:
→ Use analyze_sentiment_tool with the articles
→ This provides market sentiment analysis

ACTION 3 - IF (is ticker query OR query mentions price/chart) AND no price data:
→ Use fetch_stock_price_data tool if query involves stock prices or charts
→ This adds comprehensive price data, charts, and technical analysis

ACTION 4 - IF articles > 0 AND sentiment analyzed:
→ Provide COMPREHENSIVE ANALYSIS based on all available data
→ Include: Summary, Key Findings, Market Sentiment, Price Analysis (if available), Recommendations

IMPORTANT:
- Follow actions in order
- Don't skip sentiment analysis if articles are available
- Use fetch_stock_price_data for comprehensive price/technical analysis
- Use fetch_price_data_tool for basic historical price data only
- Provide detailed, specific insights citing articles and price data
- Each tool should only be called once when needed

What is your next action?
"""

        messages.append(HumanMessage(content=reasoning_prompt))
        
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"❌ LLM error: {type(e).__name__}")
            
            # Generate basic summary if we have data
            has_data = len(state.get('articles', [])) > 0
            
            if has_data:
                articles = state.get('articles', [])
                basic_summary = f"""
Research Results for: {query}

⚠ AI analysis partially failed

DATA COLLECTED:
- {len(articles)} articles found
- Price Data: {'Available' if has_price_data else 'Not available'}
- Sources: {', '.join(set([a.get('source', 'Unknown') for a in articles[:5]]))}

Top Headlines:
"""
                for i, article in enumerate(articles[:5], 1):
                    basic_summary += f"\n{i}. {article.get('title', 'N/A')}"
                
                return {
                    **state,
                    "final_decision": "COMPLETE",
                    "summary": basic_summary,
                    "error": f"LLM error: {type(e).__name__}",
                    "has_sufficient_data": True,
                    "iterations": iterations + 1
                }
            else:
                return {
                    **state,
                    "final_decision": "ERROR",
                    "error": f"LLM error: {type(e).__name__}",
                    "has_sufficient_data": False,
                    "iterations": iterations + 1
                }

        new_state = {
            **state,
            "messages": messages + [response],
            "iterations": iterations + 1
        }

        if response.tool_calls:
            new_state["final_decision"] = "CONTINUE"
        else:
            # Force sentiment if needed
            has_articles = len(state.get('articles', [])) > 0
            has_sentiment = bool(state.get('sentiment_report'))
            
            if has_articles and not has_sentiment:
                print(f"\n⚠ Forcing sentiment analysis...")
                articles = state.get('articles', [])
                try:
                    sentiment_result = analyze_sentiment_tool.invoke({
                        "articles_list": articles,
                        "query": query
                    })
                    if sentiment_result.get("success"):
                        state["sentiment_report"] = sentiment_result.get('sentiment_report', '')
                except Exception as e:
                    print(f"Error in forced sentiment: {type(e).__name__}")
            
            new_state["final_decision"] = "COMPLETE"
            
            # Generate final summary using the new function
            final_result = generate_final_analysis(state)
            new_state["summary"] = final_result.get("summary", "")
            new_state["has_sufficient_data"] = final_result.get("has_sufficient_data", False)

        return new_state

    def custom_tool_node(state: AgentState) -> AgentState:
        """Tool execution node."""
        messages = state["messages"]
        last_message = messages[-1]

        tool_results = []
        tools_used = state.get("tools_used", [])
        reasoning_steps = state.get("reasoning_steps", [])

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            try:
                tool_function = None
                for tool_item in tools:
                    if tool_item.name == tool_name:
                        tool_function = tool_item
                        break

                if tool_function:
                    result = tool_function.invoke(tool_args)
                else:
                    result = {"error": f"Tool {tool_name} not found", "success": False}

                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )

                tool_results.append(tool_message)
                tools_used.append(tool_name)

                # Update state based on results
                if tool_name == "intelligent_research_tool":
                    if result.get("success"):
                        state["articles"] = result.get('articles', [])
                        state["research_data"] = {
                            'query_analysis': result.get('query_analysis', {}),
                            'summary_info': result.get('summary_info', {}),
                            'search_queries_used': result.get('search_queries_used', [])
                        }
                        reasoning_steps.append(f"✓ {result.get('message')}")
                    else:
                        reasoning_steps.append(f"⚠ {result.get('message')}")

                elif tool_name == "fetch_price_data_tool":
                    if result.get("success"):
                        state["price_data"] = result.get('price_data', [])
                        reasoning_steps.append(result.get("message"))
                    else:
                        reasoning_steps.append(f"⚠ {result.get('message')}")

                elif tool_name == "fetch_stock_price_data":
                    if result.get("status") == "success":
                        state["price_data"] = result.get('price_data', [])
                        state["chart_data"] = result.get('chart_json')
                        state["technical_analysis"] = result.get('technical_indicators')
                        state["current_price"] = result.get('current_price')
                        state["currency"] = result.get('currency')
                        state["price_analysis"] = result.get('analysis')
                        reasoning_steps.append(f"✓ Comprehensive price data fetched: {result.get('message', '')}")
                    else:
                        reasoning_steps.append(f"⚠ {result.get('message')}")

                elif tool_name == "analyze_sentiment_tool":
                    if result.get("success"):
                        state["sentiment_report"] = result.get('sentiment_report', '')
                        reasoning_steps.append(result.get("message"))
                    else:
                        reasoning_steps.append(f"⚠ {result.get('message')}")

                elif tool_name == "get_stock_fundamentals":
                    if result.get("status") == "success":
                        # Store fundamentals in research_data
                        research_data = state.get("research_data", {})
                        research_data["fundamentals"] = result.get("fundamentals", {})
                        state["research_data"] = research_data
                        reasoning_steps.append(f"✓ Fundamentals data fetched")
                    else:
                        reasoning_steps.append(f"⚠ Fundamentals fetch failed: {result.get('message')}")

            except Exception as e:
                print(f"Tool error ({tool_name}): {type(e).__name__}")
                error_result = {"error": f"{type(e).__name__}", "success": False}
                tool_message = ToolMessage(
                    content=str(error_result),
                    tool_call_id=tool_call["id"]
                )
                tool_results.append(tool_message)
                reasoning_steps.append(f"✗ {tool_name} failed")

        return {
            **state,
            "messages": messages + tool_results,
            "tools_used": tools_used,
            "reasoning_steps": reasoning_steps
        }

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Determine continuation."""
        final_decision = state.get("final_decision")
        return "tools" if final_decision == "CONTINUE" else "end"

    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", custom_tool_node)
    workflow.add_node("should_fetch_price_data", should_fetch_price_data)
    workflow.add_node("fetch_price_data", fetch_price_data_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    # Add price data decision flow
    workflow.add_conditional_edges(
        "should_fetch_price_data",
        lambda x: x,
        {
            "fetch_price_data": "fetch_price_data",
            "continue_analysis": "agent"  # Go back to agent for regular analysis
        }
    )
    
    # Connect price data fetch back to agent
    workflow.add_edge("fetch_price_data", "agent")

    return workflow


_cached_react_app = None

def get_react_app():
    global _cached_react_app
    if _cached_react_app is None:
        workflow = create_react_agent()
        _cached_react_app = workflow.compile()
    return _cached_react_app


def run_universal_research(query: str, query_type: str = "auto") -> Dict:
    """Run universal financial research."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting research for: {query}")
    print(f"{'='*60}\n")
    
    try:
        initial_state = {
            "messages": [],
            "query": query,
            "query_type": query_type,
            "research_data": {},
            "articles": [],
            "price_data": [],
            "chart_data": None,
            "technical_analysis": None,
            "current_price": None,
            "currency": None,
            "price_analysis": None,
            "sentiment_report": "",
            "summary": "",
            "reasoning_steps": [],
            "tools_used": [],
            "iterations": 0,
            "max_iterations": 6,
            "final_decision": "",
            "web_results": [],
            "error": None,
            "has_sufficient_data": False,
            "should_fetch_price_data": False
        }

        react_app = get_react_app()
        final_state = react_app.invoke(initial_state)

        # Generate final analysis
        final_result = generate_final_analysis(final_state)

        return final_result

    except Exception as e:
        print(f"❌ Critical error: {type(e).__name__}")
        print(traceback.format_exc())
        
        return {
            "query": query,
            "query_type": query_type,
            "summary": f"Research failed: {type(e).__name__}",
            "articles": [],
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "has_sufficient_data": False,
            "timestamp": datetime.now().isoformat(),
            "source": "error"
        }


def run_react_analysis(ticker: str) -> Dict:
    """
    Run ReAct-pattern analysis on a stock ticker.
    
    This is the main entry point for stock analysis with the ReAct agent.
    The agent follows a Reasoning + Action loop to gather and analyze data.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
    
    Returns:
        Dictionary containing analysis results with:
        - ticker: The stock ticker analyzed
        - summary: Comprehensive analysis report
        - sentiment_report: Market sentiment analysis
        - articles: News articles found
        - price_data: Historical price data (OHLCV)
        - reasoning_steps: Steps the agent took
        - tools_used: Tools invoked during analysis
        - iterations: Number of reasoning iterations
        - timestamp: When analysis was completed
        - error: Any errors that occurred (or None)
        - has_sufficient_data: Whether enough data was gathered
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting ReAct Analysis for Stock Ticker: {ticker}")
    print(f"{'='*80}\n")
    
    try:
        # Validate and normalize ticker
        ticker = ticker.upper().strip()
        
        if not ticker:
            return {
                "ticker": ticker,
                "summary": "Invalid ticker provided",
                "sentiment_report": "",
                "articles": [],
                "price_data": [],
                "headlines": [],
                "reasoning_steps": [],
                "tools_used": [],
                "iterations": 0,
                "timestamp": datetime.now().isoformat(),
                "error": "Ticker cannot be empty",
                "has_sufficient_data": False
            }
        
        # Run universal research with ticker-specific parameters
        research_result = run_universal_research(
            query=ticker,
            query_type="auto"  # Let system auto-detect it's a ticker
        )
        
        # Ensure all required fields are present for the main.py consumer
        final_result = {
            "ticker": ticker,
            "summary": research_result.get("summary", "Research incomplete"),
            "sentiment_report": research_result.get("sentiment_report", ""),
            "articles": research_result.get("articles", []),
            "price_data": research_result.get("price_data", []),
            "chart_json": research_result.get("chart_json"),
            "technical_analysis": research_result.get("technical_analysis"),
            "current_price": research_result.get("current_price"),
            "currency": research_result.get("currency"),
            "price_analysis": research_result.get("price_analysis"),
            "headlines": [article.get("title", "") for article in research_result.get("articles", [])],
            "reasoning_steps": research_result.get("reasoning_steps", []),
            "tools_used": research_result.get("tools_used", []),
            "iterations": research_result.get("iterations", 0),
            "timestamp": research_result.get("timestamp", datetime.now().isoformat()),
            "error": research_result.get("error"),
            "has_sufficient_data": research_result.get("has_sufficient_data", False),
            "query_analysis": research_result.get("query_analysis", {}),
            "summary_info": research_result.get("summary_info", {}),
            "search_queries_used": research_result.get("search_queries_used", [])
        }
        
        print(f"\n{'='*80}")
        print(f"✅ ReAct Analysis Complete for {ticker}")
        print(f"   - Articles: {len(final_result['articles'])}")
        print(f"   - Sentiment: {'Available' if final_result['sentiment_report'] else 'Not available'}")
        print(f"   - Price Data Points: {len(final_result['price_data'])}")
        print(f"   - Chart Data: {'Available' if final_result['chart_json'] else 'Not available'}")
        print(f"   - Iterations: {final_result['iterations']}")
        print(f"{'='*80}\n")
        
        return final_result
        
    except Exception as e:
        print(f"\n❌ Critical Error in ReAct Analysis for {ticker}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)[:200]}")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        
        return {
            "ticker": ticker.upper().strip() if isinstance(ticker, str) else "",
            "summary": f"ReAct analysis failed: {type(e).__name__}",
            "sentiment_report": "",
            "articles": [],
            "price_data": [],
            "chart_json": None,
            "technical_analysis": None,
            "current_price": None,
            "currency": None,
            "price_analysis": None,
            "headlines": [],
            "reasoning_steps": [],
            "tools_used": [],
            "iterations": 0,
            "timestamp": datetime.now().isoformat(),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "has_sufficient_data": False
        }
