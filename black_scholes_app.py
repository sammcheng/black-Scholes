import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Black-Scholes formula for call and put options
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes option price.

    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying asset
    option_type : str : "call" for call option, "put" for put option

    Returns:
    float : Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculate option Greeks (Delta, Gamma, Theta, Vega)"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for both call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    if option_type == "call":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2))
    else:  # put
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    # Vega (same for both call and put)
    vega = S * np.sqrt(T) * norm.pdf(d1)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

def create_option_price_charts(S, K, T, r, volatilities):
    """Create interactive charts for option prices vs volatility"""
    call_prices = []
    put_prices = []
    
    for sigma in volatilities:
        call_prices.append(black_scholes(S, K, T, r, sigma, "call"))
        put_prices.append(black_scholes(S, K, T, r, sigma, "put"))
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Call Option Price vs Volatility', 'Put Option Price vs Volatility',
                       'Option Prices Comparison', 'Price Difference (Call - Put)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Call option chart
    fig.add_trace(
        go.Scatter(x=volatilities, y=call_prices, mode='lines', name='Call Price',
                  line=dict(color='#1f77b4', width=3), fill='tonexty'),
        row=1, col=1
    )
    
    # Put option chart
    fig.add_trace(
        go.Scatter(x=volatilities, y=put_prices, mode='lines', name='Put Price',
                  line=dict(color='#ff7f0e', width=3), fill='tonexty'),
        row=1, col=2
    )
    
    # Combined chart
    fig.add_trace(
        go.Scatter(x=volatilities, y=call_prices, mode='lines', name='Call Price',
                  line=dict(color='#1f77b4', width=3), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=volatilities, y=put_prices, mode='lines', name='Put Price',
                  line=dict(color='#ff7f0e', width=3), showlegend=False),
        row=2, col=1
    )
    
    # Price difference chart
    price_diff = [c - p for c, p in zip(call_prices, put_prices)]
    fig.add_trace(
        go.Scatter(x=volatilities, y=price_diff, mode='lines', name='Call - Put',
                  line=dict(color='#2ca02c', width=3), fill='tonexty'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Black-Scholes Option Pricing Analysis",
        showlegend=True,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Volatility (σ)", row=1, col=1)
    fig.update_xaxes(title_text="Volatility (σ)", row=1, col=2)
    fig.update_xaxes(title_text="Volatility (σ)", row=2, col=1)
    fig.update_xaxes(title_text="Volatility (σ)", row=2, col=2)
    
    fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Option Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Option Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Price Difference ($)", row=2, col=2)
    
    return fig

def create_heatmap_3d(S, K, T, r, volatilities, stock_prices):
    """Create 3D heatmap for option prices"""
    call_prices_3d = []
    put_prices_3d = []
    
    for sigma in volatilities:
        call_row = []
        put_row = []
        for S_price in stock_prices:
            call_row.append(black_scholes(S_price, K, T, r, sigma, "call"))
            put_row.append(black_scholes(S_price, K, T, r, sigma, "put"))
        call_prices_3d.append(call_row)
        put_prices_3d.append(put_row)
    
    # Create 3D surface plots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Call Option Price Surface', 'Put Option Price Surface')
    )
    
    fig.add_trace(
        go.Surface(x=stock_prices, y=volatilities, z=call_prices_3d,
                  colorscale='Blues', name='Call Price'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Surface(x=stock_prices, y=volatilities, z=put_prices_3d,
                  colorscale='Reds', name='Put Price'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="3D Option Price Surfaces",
        height=600,
        scene=dict(
            xaxis_title="Stock Price ($)",
            yaxis_title="Volatility (σ)",
            zaxis_title="Option Price ($)"
        ),
        scene2=dict(
            xaxis_title="Stock Price ($)",
            yaxis_title="Volatility (σ)",
            zaxis_title="Option Price ($)"
        )
    )
    
    return fig

def create_greeks_charts(S, K, T, r, volatilities):
    """Create charts for option Greeks"""
    greeks_data = {
        'volatility': volatilities,
        'call_delta': [], 'put_delta': [],
        'call_gamma': [], 'put_gamma': [],
        'call_theta': [], 'put_theta': [],
        'call_vega': [], 'put_vega': []
    }
    
    for sigma in volatilities:
        call_greeks = calculate_greeks(S, K, T, r, sigma, "call")
        put_greeks = calculate_greeks(S, K, T, r, sigma, "put")
        
        greeks_data['call_delta'].append(call_greeks['delta'])
        greeks_data['put_delta'].append(put_greeks['delta'])
        greeks_data['call_gamma'].append(call_greeks['gamma'])
        greeks_data['put_gamma'].append(put_greeks['gamma'])
        greeks_data['call_theta'].append(call_greeks['theta'])
        greeks_data['put_theta'].append(put_greeks['theta'])
        greeks_data['call_vega'].append(call_greeks['vega'])
        greeks_data['put_vega'].append(put_greeks['vega'])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Delta
    fig.add_trace(
        go.Scatter(x=volatilities, y=greeks_data['call_delta'], mode='lines', name='Call Delta',
                  line=dict(color='#1f77b4', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=volatilities, y=greeks_data['put_delta'], mode='lines', name='Put Delta',
                  line=dict(color='#ff7f0e', width=3)),
        row=1, col=1
    )
    
    # Gamma
    fig.add_trace(
        go.Scatter(x=volatilities, y=greeks_data['call_gamma'], mode='lines', name='Gamma',
                  line=dict(color='#2ca02c', width=3), showlegend=False),
        row=1, col=2
    )
    
    # Theta
    fig.add_trace(
        go.Scatter(x=volatilities, y=greeks_data['call_theta'], mode='lines', name='Call Theta',
                  line=dict(color='#d62728', width=3), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=volatilities, y=greeks_data['put_theta'], mode='lines', name='Put Theta',
                  line=dict(color='#9467bd', width=3), showlegend=False),
        row=2, col=1
    )
    
    # Vega
    fig.add_trace(
        go.Scatter(x=volatilities, y=greeks_data['call_vega'], mode='lines', name='Vega',
                  line=dict(color='#8c564b', width=3), showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Option Greeks Analysis",
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Volatility (σ)", row=1, col=1)
    fig.update_xaxes(title_text="Volatility (σ)", row=1, col=2)
    fig.update_xaxes(title_text="Volatility (σ)", row=2, col=1)
    fig.update_xaxes(title_text="Volatility (σ)", row=2, col=2)
    
    return fig

def create_traditional_heatmaps(S, K, T, r, volatilities):
    """Create traditional 2D heatmaps for call and put prices vs volatility"""
    call_prices = []
    put_prices = []

    # Calculate call and put prices for each volatility
    for sigma in volatilities:
        call_prices.append(black_scholes(S, K, T, r, sigma, "call"))
        put_prices.append(black_scholes(S, K, T, r, sigma, "put"))

    # Convert to numpy arrays for plotting
    call_prices = np.array(call_prices)
    put_prices = np.array(put_prices)

    # Create the heatmap figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Call option price heatmap
    ax[0].plot(volatilities, call_prices, color="#1f77b4", linewidth=3, label="Call Prices")
    ax[0].fill_between(volatilities, call_prices, alpha=0.3, color="#1f77b4")
    ax[0].set_title("Call Option Price vs Volatility", fontsize=14, fontweight='bold')
    ax[0].set_xlabel("Volatility (σ)", fontsize=12)
    ax[0].set_ylabel("Option Price ($)", fontsize=12)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=10)

    # Put option price heatmap
    ax[1].plot(volatilities, put_prices, color="#ff7f0e", linewidth=3, label="Put Prices")
    ax[1].fill_between(volatilities, put_prices, alpha=0.3, color="#ff7f0e")
    ax[1].set_title("Put Option Price vs Volatility", fontsize=14, fontweight='bold')
    ax[1].set_xlabel("Volatility (σ)", fontsize=12)
    ax[1].set_ylabel("Option Price ($)", fontsize=12)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=10)

    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_combined_heatmap(S, K, T, r, volatilities):
    """Create a combined heatmap showing both call and put prices on the same plot"""
    call_prices = []
    put_prices = []

    # Calculate call and put prices for each volatility
    for sigma in volatilities:
        call_prices.append(black_scholes(S, K, T, r, sigma, "call"))
        put_prices.append(black_scholes(S, K, T, r, sigma, "put"))

    # Convert to numpy arrays for plotting
    call_prices = np.array(call_prices)
    put_prices = np.array(put_prices)

    # Create the combined heatmap figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot both call and put prices
    ax.plot(volatilities, call_prices, color="#1f77b4", linewidth=3, label="Call Prices", marker='o', markersize=4)
    ax.plot(volatilities, put_prices, color="#ff7f0e", linewidth=3, label="Put Prices", marker='s', markersize=4)
    
    # Fill areas
    ax.fill_between(volatilities, call_prices, alpha=0.2, color="#1f77b4")
    ax.fill_between(volatilities, put_prices, alpha=0.2, color="#ff7f0e")
    
    # Add price difference line
    price_diff = call_prices - put_prices
    ax.plot(volatilities, price_diff, color="#2ca02c", linewidth=2, linestyle='--', 
            label="Call - Put Difference", alpha=0.7)

    ax.set_title("Call vs Put Option Prices vs Volatility", fontsize=16, fontweight='bold')
    ax.set_xlabel("Volatility (σ)", fontsize=14)
    ax.set_ylabel("Option Price ($)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')
    
    # Add annotations for current values
    current_call = call_prices[len(call_prices)//2]  # Middle value
    current_put = put_prices[len(put_prices)//2]
    ax.annotate(f'Call: ${current_call:.2f}', 
                xy=(volatilities[len(volatilities)//2], current_call),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f77b4', alpha=0.7),
                fontsize=10, color='white')
    
    ax.annotate(f'Put: ${current_put:.2f}', 
                xy=(volatilities[len(volatilities)//2], current_put),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff7f0e', alpha=0.7),
                fontsize=10, color='white')

    plt.tight_layout()
    
    return fig

# Streamlit app
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Black-Scholes Option Pricing Model</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Parameters")
    st.sidebar.markdown("---")
    
    # User inputs with better formatting
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        S = st.number_input("Stock Price (S)", value=100.0, step=1.0, help="Current price of the underlying asset")
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0, help="Option strike price")
        T = st.number_input("Time to Maturity (T)", value=1.0, step=0.1, help="Time to expiration in years")
    
    with col2:
        r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01, format="%.3f", help="Annual risk-free interest rate")
        sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01, format="%.3f", help="Annual volatility of the underlying asset")
    
    st.sidebar.markdown("---")
    
    # Volatility range for analysis
    vol_min = st.sidebar.slider("Min Volatility", 0.01, 0.5, 0.01, 0.01)
    vol_max = st.sidebar.slider("Max Volatility", 0.1, 1.0, 1.0, 0.01)
    volatilities = np.linspace(vol_min, vol_max, 100)
    
    # Visualization options
    st.sidebar.markdown("## 📊 Visualization Options")
    viz_type = st.sidebar.selectbox(
        "Choose Visualization Type",
        ["Interactive Charts", "Traditional Heatmaps", "Combined Heatmap", "3D Surfaces", "All Visualizations"],
        help="Select the type of charts to display"
    )
    
    # Calculate option prices
    call_price = black_scholes(S, K, T, r, sigma, option_type="call")
    put_price = black_scholes(S, K, T, r, sigma, option_type="put")
    
    # Calculate Greeks
    call_greeks = calculate_greeks(S, K, T, r, sigma, "call")
    put_greeks = calculate_greeks(S, K, T, r, sigma, "put")
    
    # Main content - Option Prices Section
    st.markdown("## 💰 Option Prices")
    st.markdown("Both call and put option prices are calculated simultaneously:")
    
    # Create two rows of metric cards for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📞 Call Option")
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #1f77b4 0%, #4a90e2 100%);">
            <div class="metric-label">Call Option Price</div>
            <div class="metric-value">${call_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Call option details
        intrinsic_call = max(0, S - K)
        time_value_call = call_price - intrinsic_call
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
            <h5>Call Option Details</h5>
            <p><strong>Intrinsic Value:</strong> ${intrinsic_call:.2f}</p>
            <p><strong>Time Value:</strong> ${time_value_call:.2f}</p>
            <p><strong>Delta:</strong> {call_greeks['delta']:.3f}</p>
            <p><strong>Gamma:</strong> {call_greeks['gamma']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📞 Put Option")
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff7f0e 0%, #ff9f40 100%);">
            <div class="metric-label">Put Option Price</div>
            <div class="metric-value">${put_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Put option details
        intrinsic_put = max(0, K - S)
        time_value_put = put_price - intrinsic_put
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #ff7f0e;">
            <h5>Put Option Details</h5>
            <p><strong>Intrinsic Value:</strong> ${intrinsic_put:.2f}</p>
            <p><strong>Time Value:</strong> ${time_value_put:.2f}</p>
            <p><strong>Delta:</strong> {put_greeks['delta']:.3f}</p>
            <p><strong>Gamma:</strong> {put_greeks['gamma']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Price comparison
    st.markdown("### 📊 Price Comparison")
    price_diff = call_price - put_price
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Call Price", f"${call_price:.2f}", f"{call_price - put_price:+.2f}")
    
    with col2:
        st.metric("Put Price", f"${put_price:.2f}", f"{put_price - call_price:+.2f}")
    
    with col3:
        st.metric("Difference (C-P)", f"${price_diff:.2f}", 
                 f"{'Call Premium' if price_diff > 0 else 'Put Premium'}")
    
    # Greeks table with better formatting
    st.markdown("## 📊 Option Greeks Comparison")
    st.markdown("Both call and put Greeks are calculated and compared:")
    
    greeks_df = pd.DataFrame({
        'Greek': ['Delta', 'Gamma', 'Theta', 'Vega'],
        'Call Option': [call_greeks['delta'], call_greeks['gamma'], call_greeks['theta'], call_greeks['vega']],
        'Put Option': [put_greeks['delta'], put_greeks['gamma'], put_greeks['theta'], put_greeks['vega']],
        'Difference': [
            call_greeks['delta'] - put_greeks['delta'],
            call_greeks['gamma'] - put_greeks['gamma'],
            call_greeks['theta'] - put_greeks['theta'],
            call_greeks['vega'] - put_greeks['vega']
        ]
    })
    greeks_df = greeks_df.round(4)
    
    # Style the dataframe
    def highlight_diff(val):
        if isinstance(val, float) and abs(val) > 0.001:
            return 'background-color: #ffeb3b'
        return ''
    
    st.dataframe(greeks_df.style.applymap(highlight_diff, subset=['Difference']), use_container_width=True)
    
    # Display visualizations based on user selection
    if viz_type == "Interactive Charts" or viz_type == "All Visualizations":
        st.markdown("## 📈 Interactive Option Price Analysis")
        st.markdown("Interactive charts showing both call and put prices across different volatilities:")
        price_chart = create_option_price_charts(S, K, T, r, volatilities)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Greeks charts
        st.markdown("## 🔍 Interactive Greeks Analysis")
        st.markdown("Greeks comparison for both call and put options:")
        greeks_chart = create_greeks_charts(S, K, T, r, volatilities)
        st.plotly_chart(greeks_chart, use_container_width=True)
    
    if viz_type == "Traditional Heatmaps" or viz_type == "All Visualizations":
        st.markdown("## 🔥 Traditional Heatmaps")
        st.markdown("Classic 2D heatmaps showing call and put option prices vs volatility:")
        heatmap_fig = create_traditional_heatmaps(S, K, T, r, volatilities)
        st.pyplot(heatmap_fig)
    
    if viz_type == "Combined Heatmap" or viz_type == "All Visualizations":
        st.markdown("## 🔥 Combined Heatmap")
        st.markdown("Single chart comparing call and put prices with price difference:")
        combined_fig = create_combined_heatmap(S, K, T, r, volatilities)
        st.pyplot(combined_fig)
    
    if viz_type == "3D Surfaces" or viz_type == "All Visualizations":
        st.markdown("## 🌊 3D Price Surfaces")
        st.markdown("3D visualization of both call and put option price surfaces:")
        stock_prices = np.linspace(S * 0.5, S * 1.5, 50)
        heatmap_3d = create_heatmap_3d(S, K, T, r, volatilities, stock_prices)
        st.plotly_chart(heatmap_3d, use_container_width=True)
    
    # Additional insights
    st.markdown("## 💡 Market Insights")
    
    # Moneyness analysis
    moneyness = S / K
    if moneyness > 1.05:
        moneyness_status = "Deep In-The-Money (ITM)"
        moneyness_color = "green"
    elif moneyness > 1.0:
        moneyness_status = "In-The-Money (ITM)"
        moneyness_color = "lightgreen"
    elif moneyness > 0.95:
        moneyness_status = "At-The-Money (ATM)"
        moneyness_color = "orange"
    elif moneyness > 0.9:
        moneyness_status = "Out-of-The-Money (OTM)"
        moneyness_color = "red"
    else:
        moneyness_status = "Deep Out-of-The-Money (OTM)"
        moneyness_color = "darkred"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: {moneyness_color}; padding: 1rem; border-radius: 10px; color: white;">
            <h4>Moneyness Analysis</h4>
            <p><strong>Ratio (S/K):</strong> {moneyness:.3f}</p>
            <p><strong>Status:</strong> {moneyness_status}</p>
            <p><strong>Call Status:</strong> {'ITM' if S > K else 'ATM' if S == K else 'OTM'}</p>
            <p><strong>Put Status:</strong> {'ITM' if S < K else 'ATM' if S == K else 'OTM'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Put-Call Parity Check
        put_call_parity = call_price - put_price - S + K * np.exp(-r * T)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;">
            <h4>Put-Call Parity Check</h4>
            <p><strong>Call - Put:</strong> ${call_price - put_price:.2f}</p>
            <p><strong>S - K*e^(-rT):</strong> ${S - K * np.exp(-r * T):.2f}</p>
            <p><strong>Parity Difference:</strong> ${put_call_parity:.4f}</p>
            <p><strong>Status:</strong> {'✅ Valid' if abs(put_call_parity) < 0.01 else '⚠️ Check'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>📊 Black-Scholes Option Pricing Model | Built with Streamlit and Plotly</p>
        <p>Both call and put options are calculated simultaneously for comprehensive analysis.</p>
        <p>This tool is for educational purposes only. Always consult with financial professionals for investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()