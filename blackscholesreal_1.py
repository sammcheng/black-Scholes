import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes-Merton Formula
def bsm_option_price(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
        #call
    call_option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    put_option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    
    return call_option_price, put_option_price

# Streamlit App
def main():
    st.title("Black-Scholes-Merton Option Pricing Model")
    st.write("This app calculates the call and put options using the Black-Scholes-Merton formula.")

    # Input parameters
    st.sidebar.header("Connect with Me:")
    linkedin_url = "https://www.linkedin.com/in/sammcheng"
    st.sidebar.markdown(
        f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;">'
        f'<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">'
        f'Samm Cheng</a>', 
        unsafe_allow_html=True
    )
    st.sidebar.header("Input Parameters")
    S = st.sidebar.number_input("Current Stock Price (S)", value=100.0, min_value=0.01)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=0.01)
    T = st.sidebar.number_input("Time to Maturity (T) in years", value=1.0, min_value=0.01)
    r = st.sidebar.number_input("Risk-Free Interest Rate (r) in %", value=5.0, min_value=0.0) / 100
    sigma = st.sidebar.number_input("Volatility (σ) in %", value=20.0, min_value=0.0) / 100

    # Calculate option price
    put_option_price = bsm_option_price(S, K, T, r, sigma, option_type)
    call_price, put_price = put_option_price

    st.success(f"The theoretical price of the Call Price option is: **{call_price:.2f}**")
    st.success(f"The theoretical price of the Put Price option is: **{put_price:.2f}**")

    

    # Heatmap Section
    st.header("Interactive Heatmap: Option Price vs. Stock Price and Time to Maturity")
    st.write("Explore how the option price varies with changes in stock price and time to maturity.")

    # Generate data for heatmap
    stock_prices = np.linspace(0.5 * S, 1.5 * S, 50)  # Range of stock prices (50% to 150% of S)
    times_to_maturity = np.linspace(0.1, T, 50)  # Time to maturity from 0.1 years to T years
    heatmap_data = []

    for s in stock_prices:
        for t in times_to_maturity:
            # Calculate call and put prices
            call_price = bsm_option_price(s, K, t, r, sigma, 'call')
            put_price = bsm_option_price(s, K, t, r, sigma, 'put')
            heatmap_data.append([s, t, call_price, put_price])

    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, columns=["Stock Price", "Time to Maturity", "Call Price", "Put Price"])
    heatmap_df2 = pd.DataFrame(heatmap_data, columns=["Stock Price", "Time to Maturity", "Call Price", "Put Price"])
    # Reshape data for heatmap
    
    heatmap_pivot = heatmap_df.pivot(index="Stock Price", columns="Time to Maturity", values="Call Price")
    title = "Call Option Price Heatmap"
    
    heatmap_pivot_2 = heatmap_df2.pivot(index="Stock Price", columns="Time to Maturity", values="Put Price")
    title_2 = "Put Option Price Heatmap"

    # Create heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_pivot, cmap="RdYlGn", annot=False, fmt="1.2f", cbar_kws={"label": "Call Price"})
    plt.title(title)
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Stock Price")
    st.pyplot(plt)

    # second heatmap for put 
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_pivot_2, cmap="RdYlGn", annot=False, fmt="1.2f", cbar_kws={"label": "Put Price"})
    plt.title(title_2)
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Stock Price")
    st.pyplot(plt)

    # Display formula and explanation
    st.header("Black-Scholes-Merton Formula")
    st.latex(r"""
    \text{Call Option Price} = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
    """)
    st.latex(r"""
    \text{Put Option Price} = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
    """)
    st.latex(r"""
    d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}}
    """)
    st.latex(r"""
    d_2 = d_1 - \sigma \sqrt{T}
    """)

# Run the app
if __name__ == "__main__":
    main()
