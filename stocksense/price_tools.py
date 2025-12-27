# Add to your react_agent.py or create a new file tools/price_tools.py

from typing import Dict, Any, Optional
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import numpy as np


class PriceDataTools:
    """Tools for fetching and analyzing stock price data."""
    
    @staticmethod
    def fetch_price_data(ticker: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Fetch comprehensive price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            Dictionary with price data and analysis
        """
        try:
            print(f"[Price Tool] Fetching price data for {ticker} ({period})...")
            
            # Fetch data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {
                    "tool": "fetch_price_data",
                    "status": "error",
                    "message": f"No price data found for {ticker}",
                    "ticker": ticker
                }
            
            # Get current info
            info = stock.info
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            currency = info.get('currency', 'USD')
            
            # Convert to list format
            price_data = []
            for date, row in hist.iterrows():
                price_data.append({
                    "Date": date.strftime('%Y-%m-%d'),
                    "Open": float(row['Open']),
                    "High": float(row['High']),
                    "Low": float(row['Low']),
                    "Close": float(row['Close']),
                    "Volume": int(row['Volume']),
                    "Adj_Close": float(row['Close'])
                })
            
            # Calculate metrics
            df = hist.copy()
            
            # Price change
            price_change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            
            # Moving averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            df['RSI'] = PriceDataTools.calculate_rsi(df['Close'])
            
            # Volume metrics
            avg_volume = int(df['Volume'].mean())
            latest_volume = int(df['Volume'].iloc[-1])
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility
            daily_returns = df['Close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Support/Resistance levels (simplified)
            support = df['Low'].tail(20).min()
            resistance = df['High'].tail(20).max()
            
            # Generate chart JSON
            chart_json = PriceDataTools.generate_price_chart(ticker, df)
            
            return {
                "tool": "fetch_price_data",
                "status": "success",
                "ticker": ticker,
                "period": period,
                "current_price": current_price,
                "currency": currency,
                "price_change_pct": float(price_change_pct),
                "price_data_summary": {
                    "days": len(df),
                    "start_date": df.index[0].strftime('%Y-%m-%d'),
                    "end_date": df.index[-1].strftime('%Y-%m-%d'),
                    "high": float(df['High'].max()),
                    "low": float(df['Low'].min()),
                    "avg_volume": avg_volume,
                    "latest_volume": latest_volume,
                    "volume_ratio": float(volume_ratio)
                },
                "technical_indicators": {
                    "ma20": float(df['MA20'].iloc[-1]) if not pd.isna(df['MA20'].iloc[-1]) else None,
                    "ma50": float(df['MA50'].iloc[-1]) if not pd.isna(df['MA50'].iloc[-1]) else None,
                    "ma200": float(df['MA200'].iloc[-1]) if not pd.isna(df['MA200'].iloc[-1]) else None,
                    "rsi": float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None,
                    "support": float(support),
                    "resistance": float(resistance),
                    "volatility_pct": float(volatility)
                },
                "price_data": price_data[-50:],  # Last 50 days for display
                "chart_json": chart_json,
                "analysis": PriceDataTools.generate_technical_analysis(df, current_price)
            }
            
        except Exception as e:
            return {
                "tool": "fetch_price_data",
                "status": "error",
                "message": f"Error fetching price data: {str(e)}",
                "ticker": ticker
            }
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def generate_price_chart(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate Plotly chart JSON."""
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f'{ticker} Price Chart', 'Volume', 'RSI'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            if 'MA20' in df.columns and not df['MA20'].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA20'],
                        name='MA20',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'MA50' in df.columns and not df['MA50'].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA50'],
                        name='MA50',
                        line=dict(color='red', width=1)
                    ),
                    row=1, col=1
                )
            
            # Volume chart with color coding
            colors = ['red' if row['Close'] < row['Open'] else 'green' 
                     for _, row in df.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # RSI chart
            if 'RSI' in df.columns and not df['RSI'].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=2),
                        showlegend=False
                    ),
                    row=3, col=1
                )
                
                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             opacity=0.5, row=3, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                             opacity=0.3, row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Technical Analysis',
                height=700,
                xaxis_rangeslider_visible=False,
                template='plotly_white',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            # Convert to JSON
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"[Price Tool] Chart generation error: {str(e)}")
            return {}
    
    @staticmethod
    def generate_technical_analysis(df: pd.DataFrame, current_price: float) -> Dict[str, str]:
        """Generate technical analysis insights."""
        analysis = {
            "trend": "neutral",
            "momentum": "neutral",
            "volume": "normal",
            "key_levels": "",
            "summary": ""
        }
        
        try:
            # Trend analysis
            if 'MA20' in df.columns and 'MA50' in df.columns:
                ma20 = df['MA20'].iloc[-1]
                ma50 = df['MA50'].iloc[-1]
                
                if not pd.isna(ma20) and not pd.isna(ma50):
                    if current_price > ma20 > ma50:
                        analysis["trend"] = "strong_uptrend"
                    elif current_price > ma20 and ma20 > ma50:
                        analysis["trend"] = "uptrend"
                    elif current_price < ma20 < ma50:
                        analysis["trend"] = "strong_downtrend"
                    elif current_price < ma20 and ma20 < ma50:
                        analysis["trend"] = "downtrend"
                    else:
                        analysis["trend"] = "consolidation"
            
            # Momentum analysis (RSI)
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if not pd.isna(rsi):
                    if rsi > 70:
                        analysis["momentum"] = "overbought"
                    elif rsi < 30:
                        analysis["momentum"] = "oversold"
                    elif rsi > 50:
                        analysis["momentum"] = "bullish"
                    else:
                        analysis["momentum"] = "bearish"
            
            # Volume analysis
            avg_volume = df['Volume'].mean()
            latest_volume = df['Volume'].iloc[-1]
            if latest_volume > avg_volume * 1.5:
                analysis["volume"] = "high"
            elif latest_volume < avg_volume * 0.5:
                analysis["volume"] = "low"
            
            # Support/Resistance
            support = df['Low'].tail(20).min()
            resistance = df['High'].tail(20).max()
            analysis["key_levels"] = f"Support: ${support:.2f}, Resistance: ${resistance:.2f}"
            
            # Generate summary
            trend_map = {
                "strong_uptrend": "in a strong uptrend",
                "uptrend": "in an uptrend",
                "downtrend": "in a downtrend",
                "strong_downtrend": "in a strong downtrend",
                "consolidation": "consolidating",
                "neutral": "showing neutral trend"
            }
            
            momentum_map = {
                "overbought": "overbought conditions",
                "oversold": "oversold conditions",
                "bullish": "bullish momentum",
                "bearish": "bearish momentum",
                "neutral": "neutral momentum"
            }
            
            volume_map = {
                "high": "on high volume",
                "low": "on low volume",
                "normal": "on normal volume"
            }
            
            analysis["summary"] = (
                f"The stock is {trend_map.get(analysis['trend'], 'trending')} "
                f"with {momentum_map.get(analysis['momentum'], 'momentum')} "
                f"{volume_map.get(analysis['volume'], '')}. "
                f"{analysis['key_levels']}"
            )
            
        except Exception as e:
            analysis["summary"] = f"Technical analysis limited: {str(e)}"
        
        return analysis
    
    @staticmethod
    def compare_tickers(ticker1: str, ticker2: str, period: str = "6mo") -> Dict[str, Any]:
        """Compare two stocks."""
        # Implementation for comparison
        pass