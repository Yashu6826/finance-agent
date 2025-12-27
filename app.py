import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
from stocksense.database import init_db
from stocksense.data_collectors import get_stock_currency, get_stock_info
PLOTLY_AVAILABLE = True

st.set_page_config(
    page_title="FinanceBot AI Research Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }

    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .analysis-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
        color: #333333;
    }

    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
        color: #333333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .query-chip {
        display: inline-block;
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #e1e5e9;
    }

    .query-chip:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }

    .status-online {
        color: #28a745;
        font-weight: bold;
    }

    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e1e5e9;
        padding: 0.75rem;
        font-size: 16px;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stProgress .st-bo {
        background-color: #667eea;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    .query-type-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    .badge-ticker {
        background: #e3f2fd;
        color: #1976d2;
    }

    .badge-general {
        background: #f3e5f5;
        color: #7b1fa2;
    }

    .badge-news {
        background: #fff3e0;
        color: #e65100;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'backend_status' not in st.session_state:
        st.session_state.backend_status = None
    if 'selected_query' not in st.session_state:
        st.session_state.selected_query = ""
    if 'query_type' not in st.session_state:
        st.session_state.query_type = "auto"

initialize_session_state()

@st.cache_data
def load_company_ticker_mapping():
    """Load company name -> ticker mapping from minimal CSV (Symbol, Name)."""
    csv_path = Path(__file__).parent / "nasdaq_screener.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=["Symbol", "Name"])
            df['Symbol'] = df['Symbol'].astype(str).str.strip()
            df = df.dropna(subset=["Symbol", "Name"]).drop_duplicates(subset=["Symbol"])
            mapping = dict(zip(df['Name'], df['Symbol']))
            return mapping, df
        except Exception as e:
            st.warning(f"Error loading CSV file: {e}")
    fallback_mapping = {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "Alphabet Inc.": "GOOGL",
        "Amazon.com Inc.": "AMZN",
        "Tesla Inc.": "TSLA",
        "NVIDIA Corporation": "NVDA",
        "Meta Platforms Inc.": "META",
        "Netflix Inc.": "NFLX",
        "Advanced Micro Devices Inc.": "AMD",
        "Intel Corporation": "INTC"
    }
    return fallback_mapping, None

def create_candlestick_chart(price_data: list, currency: str = "USD") -> Optional[go.Figure]:
    """Create an interactive candlestick chart with moving averages."""
    if not price_data:
        return None
    
    try:
        df = pd.DataFrame(price_data)
        
        # Ensure Date column exists
        if 'Date' not in df.columns:
            st.error("Missing 'Date' column in price data")
            return None
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        if df.empty:
            st.error("No valid date data found")
            return None
        
        df = df.sort_values('Date')
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing '{col}' column in price data")
                return None
        
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    except KeyError as e:
        st.error(f"Error creating chart: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing price data: {str(e)}")
        return None
    
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"Stock Price with Moving Averages - {currency}",
            yaxis_title=f"Price ({currency})",
            xaxis_title="Date",
            template="plotly_white",
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart visualization: {str(e)}")
        return None

try:
    BACKEND_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
except:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def check_backend_status() -> bool:
    """Check if the backend is online."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        status = response.status_code == 200
        st.session_state.backend_status = status
        return status
    except:
        st.session_state.backend_status = False
        return False

def create_styled_header():
    try:
        import streamlit.components.v1 as components

        header_html = (
            '<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.6rem 1rem; border-radius: 10px; margin-bottom: 1.6rem; position: relative;">'
            '<div style="position: absolute; top: 0.8rem; right: 0.8rem; display: flex; gap: 0.45rem;">'
            '<a href="https://github.com/spkap" target="_blank" rel="noopener noreferrer" '
            'title="spkap — GitHub profile" aria-label="GitHub profile (spkap)" role="link" tabindex="0" '
            'style="display:inline-flex; align-items:center; justify-content:center; width:40px; height:40px; padding:6px; border-radius:20px; background: rgba(255,255,255,0.02); transition: transform 0.12s ease, background 0.12s ease;" '
            'onmouseover="this.style.background=\'rgba(255,255,255,0.12)\'; this.style.transform=\'scale(1.06)\'" '
            'onmouseout="this.style.background=\'rgba(255,255,255,0.02)\'; this.style.transform=\'none\'">'
            '<svg width="28" height="28" viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            '<defs><linearGradient id="profgrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#1E3A8A"/><stop offset="100%" stop-color="#1E3A8A"/></linearGradient></defs>'
            '<circle cx="18" cy="18" r="16" fill="url(#profgrad)" />'
            '<g fill="#ffffff" transform="translate(0,0)" >'
            '<circle cx="18" cy="13" r="4" />'
            '<path d="M10 25c1-4 5-6 8-6s7 2 8 6" />'
            '</g>'
            '</svg></a>'
            '<a href="https://github.com/Yashu6826/finance-agent" target="_blank" rel="noopener noreferrer" '
            'title="FinanceBot-AI — repository" aria-label="Repository FinanceBot-AI" role="link" tabindex="0" '
            'style="display:inline-flex; align-items:center; justify-content:center; width:40px; height:40px; padding:6px; border-radius:8px; background: rgba(255,255,255,0.02); transition: transform 0.12s ease, background 0.12s ease;" '
            'onmouseover="this.style.background=\'rgba(255,255,255,0.12)\'; this.style.transform=\'scale(1.06)\'" '
            'onmouseout="this.style.background=\'rgba(255,255,255,0.02)\'; this.style.transform=\'none\'">'
            '<svg width="20" height="20" viewBox="0 0 16 16" fill="white" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />'
            '</svg></a>'
            '</div>'
            '<h1 style="color: white; text-align: center; margin: 0; font-size: 1.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📈 FinanceBot AI Research Agent</h1>'
            '<p style="color: #f0f0f0; text-align: center; margin: 0.35rem 0 0 0; font-size: 1rem; opacity: 0.92;">Universal Financial Research Using Reasoning & Action</p>'
            '</div>'
        )

        components.html(header_html, height=120)
    except Exception:
        st.markdown(
            '<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.6rem 1rem; border-radius: 10px; margin-bottom: 1.6rem;">'
            '<h1 style="color: white; text-align: center; margin: 0; font-size: 1.5rem;">📈 FinanceBot AI Research Agent</h1>'
            '<p style="color: #f0f0f0; text-align: center; margin: 0.35rem 0 0 0; font-size: 1rem;">Universal Financial Research Using Reasoning & Action</p>'
            '</div>',
            unsafe_allow_html=True,
        )

def display_hero_section():
    create_styled_header()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if check_backend_status():
            st.success("🟢 Backend Connected & Ready", icon="✅")
        else:
            st.error("🔴 Backend Connection Failed", icon="❌")

    st.markdown("<br>", unsafe_allow_html=True)

def detect_query_type(query: str) -> str:
    """Detect query type for UI display."""
    import re
    query = query.strip()
    
    # Check if it's a ticker (1-5 uppercase letters)
    ticker_pattern = r'^[A-Z]{1,5}(?:\.[A-Z]{2})?$'
    if re.match(ticker_pattern, query.upper()):
        return "ticker"
    
    # Check if it contains ticker-like patterns
    if re.search(r'\b[A-Z]{2,5}\b', query):
        return "ticker"
    
    # Check for news/trend keywords
    news_keywords = ['news', 'latest', 'recent', 'trend', 'market', 'analysis']
    if any(keyword in query.lower() for keyword in news_keywords):
        return "news"
    
    return "general"

def display_query_input():
    """Enhanced input section supporting both tickers and general queries."""
    st.markdown("### 🔍 What would you like to research?")
    
    # Query type selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Enter your research query:**")
    
    with col2:
        query_mode = st.selectbox(
            "Mode",
            options=["Auto Detect", "Stock Ticker", "General Query"],
            index=0,
            help="Auto Detect will analyze your input automatically",
            label_visibility="collapsed"
        )
    
    # Main query input
    query_placeholder = {
        "Auto Detect": "e.g., AAPL, Tesla earnings, semiconductor market trends...",
        "Stock Ticker": "e.g., AAPL, TSLA, GOOGL",
        "General Query": "e.g., What are the latest AI chip developments?"
    }
    
    user_query = st.text_area(
        "Research Query",
        value=st.session_state.selected_query,
        placeholder=query_placeholder[query_mode],
        height=100,
        help="Enter a stock ticker, company name, or any financial research question",
        label_visibility="collapsed"
    )
    
    if user_query and user_query != st.session_state.selected_query:
        st.session_state.selected_query = user_query.strip()
    
    # Example queries
    st.markdown("**💡 Example Queries:**")
    
    example_queries = {
        "Stock Tickers": [
            "AAPL",
            "TSLA",
            "NVDA",
            "MSFT"
        ],
        "Company Research": [
            "Apple latest earnings",
            "Tesla production updates",
            "Microsoft AI strategy"
        ],
        "Market Trends": [
            "semiconductor market trends",
            "AI chip developments",
            "renewable energy stocks outlook"
        ],
        "News & Events": [
            "latest tech IPOs",
            "Fed interest rate impact",
            "cryptocurrency market analysis"
        ]
    }
    
    tabs = st.tabs(list(example_queries.keys()))
    
    for tab, (category, queries) in zip(tabs, example_queries.items()):
        with tab:
            cols = st.columns(len(queries))
            for col, query in zip(cols, queries):
                with col:
                    if st.button(query, key=f"example_{query}", use_container_width=True):
                        st.session_state.selected_query = query
                        st.rerun()
    
    # Display detected query type
    if st.session_state.selected_query:
        detected_type = detect_query_type(st.session_state.selected_query)
        type_badges = {
            "ticker": '<span class="query-type-badge badge-ticker">📊 Stock Ticker</span>',
            "news": '<span class="query-type-badge badge-news">📰 News/Trends</span>',
            "general": '<span class="query-type-badge badge-general">🔍 General Research</span>'
        }
        
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px; text-align: center;">
            <span style="color: #666;">Detected:</span> {type_badges.get(detected_type, type_badges["general"])}
        </div>
        """, unsafe_allow_html=True)
    
    # Quick select for popular stocks
    with st.expander("📈 Quick Select Popular Stocks", expanded=False):
        st.markdown("**Tech Giants:**")
        cols = st.columns(5)
        tech_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        for i, ticker in enumerate(tech_tickers):
            with cols[i]:
                if st.button(ticker, key=f"tech_{ticker}", use_container_width=True):
                    st.session_state.selected_query = ticker
                    st.rerun()
        
        st.markdown("**AI & Semiconductors:**")
        cols = st.columns(5)
        ai_tickers = ["NVDA", "AMD", "INTC", "TSM", "AVGO"]
        
        for i, ticker in enumerate(ai_tickers):
            with cols[i]:
                if st.button(ticker, key=f"ai_{ticker}", use_container_width=True):
                    st.session_state.selected_query = ticker
                    st.rerun()
        
        st.markdown("**Other Popular:**")
        cols = st.columns(5)
        other_tickers = ["TSLA", "NFLX", "DIS", "V", "JPM"]
        
        for i, ticker in enumerate(other_tickers):
            with cols[i]:
                if st.button(ticker, key=f"other_{ticker}", use_container_width=True):
                    st.session_state.selected_query = ticker
                    st.rerun()
    
    return st.session_state.selected_query

def validate_query(query: str) -> tuple[bool, str]:
    """Validate query input."""
    if not query or not query.strip():
        return False, "Please enter a research query"
    
    if len(query.strip()) < 2:
        return False, "Query too short - please provide more detail"
    
    return True, ""

def trigger_research(query: str) -> Optional[Dict[str, Any]]:
    """Triggers universal research via backend API."""
    try:
        if not check_backend_status():
            st.error("🔌 Backend server is offline. Please start the FastAPI server.")
            return None

        with st.spinner("🤖 ReAct Agent is researching..."):
            # Use the universal research endpoint
            response = requests.post(
                f"{BACKEND_URL}/research",
                json={"query": query, "query_type": "auto"},
                timeout=360
            )

        if response.status_code == 200:
            result = response.json()
            
            analysis_data = result.get('data', {})
            
            if analysis_data.get('source') == 'react_analysis':
                st.success("✅ Fresh research completed!")
            elif analysis_data.get('source') == 'cache':
                st.info("📚 Retrieved from cache")
            
            result_obj = {
                'query': query,
                'data': analysis_data,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }

            st.session_state.analysis_result = result_obj
            st.session_state.analysis_history.insert(0, result_obj)
            if len(st.session_state.analysis_history) > 10:
                st.session_state.analysis_history.pop()

            return result_obj

        else:
            st.error(f"❌ Research failed: Status {response.status_code}")
            if response.text:
                st.error(f"Details: {response.text}")
            return None

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Research may still be processing.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend. Please ensure the server is running.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def display_research_results(data: Dict[str, Any], query: str):
    """Display universal research results with enhanced formatting."""
    
    # Detect query type for better display
    query_analysis = data.get('query_analysis', {})
    is_ticker = query_analysis.get('is_ticker', False)
    ticker = query_analysis.get('ticker', '')
    
    # Create tabs
    if is_ticker and ticker:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Price Analysis", "📰 News & Sentiment", "⚙️ Agent Process"])
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Research Summary", "📰 Articles & Sources", "⚙️ Agent Process"])
    
    with tab1:
        if is_ticker and ticker:
            st.markdown(f"### 📊 Overview - {ticker}")
            
            # Display currency and exchange info
            try:
                currency = data.get('currency', 'USD')
                current_price = data.get('current_price')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"💱 **Currency:** {currency}")
                with col2:
                    if current_price:
                        st.metric("Current Price", f"{currency} {current_price:.2f}")
                with col3:
                    price_change = data.get('price_analysis', {}).get('price_change_pct', 0)
                    st.metric("Change", f"{price_change:+.2f}%")
            except Exception as e:
                pass
        else:
            st.markdown(f"### 📊 Research Summary")
            st.markdown(f"**Query:** {query}")
        
        # Display main summary
        summary = data.get('summary', 'Research completed successfully')
        
        if summary and summary != "Research completed successfully":
            st.markdown(f"""
            <div class="analysis-card fade-in">
                <h4 style="color: #667eea; margin-bottom: 1rem; font-size: 1.3rem; border-bottom: 2px solid #e1e5e9; padding-bottom: 0.5rem;">
                    📝 Analysis
                </h4>
                <div>{summary}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📊 Detailed analysis summary not available")
        
        # Display key findings
        summary_info = data.get('summary_info', {})
        if summary_info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                article_count = summary_info.get('total_articles', 0)
                st.metric("📰 Articles Found", article_count)
            
            with col2:
                source_count = summary_info.get('unique_sources', 0)
                st.metric("📚 Sources", source_count)
            
            with col3:
                if summary_info.get('date_range'):
                    st.metric("📅 Date Range", summary_info['date_range'])
    
    # Tab 2 - Price Analysis (only for tickers) or Articles (for general queries)
    if is_ticker and ticker:
        with tab2:
            st.markdown("### 📈 Price Analysis")
            
            # Display candlestick chart
            price_data = data.get('price_data', [])
            if price_data and len(price_data) > 0:
                try:
                    currency = data.get('currency', 'USD')
                    fig = create_candlestick_chart(price_data, currency)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"📊 Displaying {len(price_data)} days of price data")
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
            else:
                st.info("📈 No historical price data available. Run analysis to fetch price data.")
            
            # Display technical analysis
            tech_analysis = data.get('technical_analysis', {})
            if tech_analysis:
                st.markdown("#### 🔬 Technical Indicators")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'rsi' in tech_analysis:
                        rsi = tech_analysis['rsi']
                        rsi_status = "Overbought ⚠️" if rsi > 70 else "Oversold ✅" if rsi < 30 else "Neutral"
                        st.metric("RSI", f"{rsi:.1f}", rsi_status)
                
                with col2:
                    if 'ma20' in tech_analysis:
                        st.metric("MA20", f"{tech_analysis['ma20']:.2f}")
                
                with col3:
                    if 'ma50' in tech_analysis:
                        st.metric("MA50", f"{tech_analysis['ma50']:.2f}")
                
                with col4:
                    if 'volume' in tech_analysis:
                        st.metric("Volume", f"{tech_analysis['volume']:,.0f}")
    else:
        with tab2:
            st.markdown("### 📰 Articles & Sources")
            
            articles = data.get('articles', [])
            if articles:
                st.markdown(f"**Found {len(articles)} relevant articles**")
                
                for i, article in enumerate(articles[:10], 1):
                    with st.expander(f"📄 {i}. {article.get('title', 'Untitled')[:80]}..."):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Title:** {article.get('title', 'N/A')}")
                            st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
                            
                            description = article.get('description') or article.get('content_snippet', '')
                            if description:
                                st.markdown(f"**Summary:** {description[:300]}...")
                        
                        with col2:
                            published = article.get('published_at', 'N/A')
                            st.markdown(f"**Published:** {published}")
                            
                            if article.get('url'):
                                # Continuation of the display_research_results function and remaining code

                                st.markdown(f"[Read Full Article]({article['url']})")
            else:
                st.info("No articles found for this query")
    
    # Tab 3 (or 2 for non-ticker) - News & Sentiment
    sentiment_tab = tab3 if is_ticker else tab2
    
    with sentiment_tab:
        st.markdown("### 📰 News & Sentiment Analysis")
        
        sentiment_report = data.get('sentiment_report', '')
        
        if sentiment_report:
            st.markdown("""
                <h4 style="color: #667eea; margin-bottom: 1rem; font-size: 1.3rem; border-bottom: 2px solid #e1e5e9; padding-bottom: 0.5rem;">
                    📊 Market Sentiment Report
                </h4>
            """, unsafe_allow_html=True)
            
            # Try to parse structured sentiment
            if isinstance(sentiment_report, str):
                try:
                    sentiment_data = json.loads(sentiment_report)
                    if isinstance(sentiment_data, list):
                        for i, item in enumerate(sentiment_data, 1):
                            headline = item.get('headline', 'N/A')
                            sentiment = item.get('sentiment', 'N/A')
                            justification = item.get('justification', 'N/A')
                            
                            with st.expander(f"📰 Article {i}: {headline[:80]}..."):
                                st.markdown(f"**Headline:** {headline}")
                                st.markdown(f"**Sentiment:** {sentiment}")
                                st.markdown(f"**Analysis:** {justification}")
                    else:
                        st.markdown(sentiment_report)
                except json.JSONDecodeError:
                    st.markdown(sentiment_report)
            elif isinstance(sentiment_report, list):
                for i, item in enumerate(sentiment_report, 1):
                    headline = item.get('headline', 'N/A')
                    sentiment = item.get('sentiment', 'N/A')
                    justification = item.get('justification', 'N/A')
                    
                    with st.expander(f"📰 Article {i}: {headline[:80]}..."):
                        st.markdown(f"**Headline:** {headline}")
                        st.markdown(f"**Sentiment:** {sentiment}")
                        st.markdown(f"**Analysis:** {justification}")
            else:
                st.markdown(str(sentiment_report))
        else:
            st.info("No sentiment analysis available for this query")
    
    # Last tab - Agent Process
    agent_tab = tab4 if is_ticker else tab3
    
    with agent_tab:
        st.markdown("### ⚙️ Agent Reasoning Process")
        
        reasoning_steps = data.get('reasoning_steps', [])
        tools_used = data.get('tools_used', [])
        iterations = data.get('iterations', 0)
        is_cached = data.get('source') == 'cache'
        
        has_meaningful_data = (reasoning_steps and not is_cached) or (tools_used and not is_cached)
        
        if has_meaningful_data:
            st.markdown("""
                <h4 style="color: #667eea; margin-bottom: 1rem; font-size: 1.3rem; border-bottom: 2px solid #e1e5e9; padding-bottom: 0.5rem;">
                    🧠 ReAct Agent Decision Process
                </h4>
            """, unsafe_allow_html=True)
            
            # Display reasoning steps
            if reasoning_steps and not is_cached:
                st.markdown("**📋 Reasoning Steps:**")
                for i, step in enumerate(reasoning_steps, 1):
                    st.markdown(f"{i}. {step}")
            
            # Display tools used
            if tools_used and not is_cached:
                st.markdown("**🔧 Tools Used:**")
                unique_tools = list(set(tools_used))
                for tool in unique_tools:
                    st.markdown(f"• {tool}")
            
            # Display iterations
            if iterations and not is_cached:
                st.markdown(f"**🔄 Analysis Iterations:** {iterations}")
            
            # Display search queries used
            search_queries = data.get('search_queries_used', [])
            if search_queries:
                st.markdown("**🔍 Search Queries Used:**")
                for i, sq in enumerate(search_queries, 1):
                    st.markdown(f"{i}. `{sq}`")
        else:
            if is_cached:
                st.info("📚 This is a cached result. Reasoning steps are not available for cached results.")
            else:
                st.warning("⚠️ No detailed reasoning information available.")
        
        # Debug info
        with st.expander("🔍 Raw Research Data (Debug)"):
            st.json(data)

def display_analysis_history():
    """Displays analysis history in sidebar."""
    if st.session_state.analysis_history:
        st.markdown("### 📚 Recent Research")

        for i, analysis in enumerate(st.session_state.analysis_history):
            query = analysis.get('query', 'Unknown')
            timestamp = analysis['timestamp']
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%m/%d %H:%M")

            # Truncate long queries
            display_query = query[:30] + "..." if len(query) > 30 else query

            if st.button(f"📊 {display_query} - {time_str}", key=f"history_{i}"):
                st.session_state.analysis_result = analysis
                st.rerun()

def display_sidebar():
    with st.sidebar:
        st.markdown("### 🔧 System Status")

        status = check_backend_status()
        if status:
            st.markdown('<p class="status-online">🟢 Backend Online</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-offline">🔴 Backend Offline</p>', unsafe_allow_html=True)
            st.warning("Start the FastAPI server to enable research")

        display_analysis_history()

        st.markdown("---")

        st.markdown("### ℹ️ About Universal Research")

        with st.expander("How it works"):
            st.markdown("""
            **ReAct Pattern (Reasoning + Action):**

            1. 🧠 **Analyzes** your query intelligently
            2. 🔍 **Detects** whether it's a ticker, company, or trend
            3. 🔧 **Selects** appropriate research tools
            4. 📊 **Gathers** news, prices, and data
            5. 🤖 **Analyzes** sentiment and trends
            6. ✅ **Delivers** comprehensive insights

            **Query Types Supported:**
            - Stock tickers (AAPL, TSLA)
            - Company research queries
            - Market trend analysis
            - News and event research
            - General financial questions
            """)

        with st.expander("Data Sources"):
            st.markdown("""
            - 📰 **NewsAPI**: Latest financial news
            - 📈 **Yahoo Finance**: Real-time prices
            - 🤖 **Google Gemini**: AI analysis
            - 💾 **SQLite**: Intelligent caching
            """)

        with st.expander("Example Queries"):
            st.markdown("""
            **Stock Analysis:**
            - `AAPL`
            - `TSLA earnings analysis`
            
            **Market Trends:**
            - `semiconductor market outlook`
            - `AI chip developments`
            
            **News Research:**
            - `latest tech IPOs`
            - `Fed interest rate impact`
            
            **Company Research:**
            - `Microsoft AI strategy`
            - `Tesla production updates`
            """)

        st.markdown("---")
        
        if st.button("🗑️ Clear Session Data", help="Clear current results and history"):
            st.session_state.analysis_result = None
            st.session_state.analysis_history = []
            st.session_state.selected_query = ""
            st.rerun()

def main():
    """Main function for the Streamlit application."""
    init_db()
    display_hero_section()
    display_sidebar()

    main_container = st.container()

    with main_container:
        # Universal query input
        query = display_query_input()

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            is_valid, error_msg = validate_query(query)

            if not is_valid and query:
                st.error(f"❌ {error_msg}")

            research_button = st.button(
                "🚀 Start Research",
                type="primary",
                use_container_width=True,
                disabled=not is_valid,
                help="Trigger AI-powered research using the ReAct pattern"
            )

            if research_button and is_valid:
                result = trigger_research(query)
                if result and result.get('success'):
                    st.success(f"✅ Research completed!")

        st.markdown("---")

        # Display results
        if st.session_state.analysis_result:
            result_data = st.session_state.analysis_result
            query = result_data['query']
            data = result_data['data']

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"## 📊 Research Results")
                
                # Display query with badge
                query_type = detect_query_type(query)
                type_badges = {
                    "ticker": '🎯 Stock Analysis',
                    "news": '📰 News Research',
                    "general": '🔍 General Research'
                }
                st.markdown(f"**{type_badges.get(query_type, '🔍 Research')}:** {query}")

            with col2:
                if st.button("🗑️ Clear", help="Clear current results"):
                    st.session_state.analysis_result = None
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # Display results with enhanced formatting
            display_research_results(data, query)

        else:
            # Welcome screen
            st.markdown("""
            <div class="analysis-card fade-in" style="text-align: center; padding: 3rem;">
                <h3>👋 Welcome to FinanceBot Universal Research</h3>
                <p style="font-size: 1.1rem; color: #666; margin: 1.5rem 0;">
                    Ask me anything about stocks, markets, or financial news
                </p>
                <div style="margin-top: 2rem;">
                    <p style="color: #888; font-size: 0.95rem;">
                        🎯 Analyze specific stocks<br>
                        📰 Research market trends<br>
                        🔍 Get latest financial news<br>
                        🤖 AI-powered insights
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show example queries as cards
            st.markdown("### 💡 Try These Examples:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="analysis-card" style="padding: 1.5rem; min-height: 150px;">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">📊 Stock Analysis</h4>
                    <p style="color: #666; font-size: 0.9rem;">Get comprehensive analysis of any stock</p>
                    <code style="background: #f0f2f6; padding: 0.3rem 0.5rem; border-radius: 4px;">AAPL</code>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="analysis-card" style="padding: 1.5rem; min-height: 150px;">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">🔍 Market Research</h4>
                    <p style="color: #666; font-size: 0.9rem;">Explore trends and developments</p>
                    <code style="background: #f0f2f6; padding: 0.3rem 0.5rem; border-radius: 4px;">AI chip market</code>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="analysis-card" style="padding: 1.5rem; min-height: 150px;">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">📰 News Analysis</h4>
                    <p style="color: #666; font-size: 0.9rem;">Latest updates and sentiment</p>
                    <code style="background: #f0f2f6; padding: 0.3rem 0.5rem; border-radius: 4px;">Tesla earnings</code>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()