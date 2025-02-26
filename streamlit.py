import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Streamlit App
def main():
    st.title("Black-Scholes-Merton Option Pricing Model")
    st.write("Calculate the theoretical prices of call and put options using the Black-Scholes-Merton formula.")

    # Sidebar for user inputs
    st.sidebar.header("linkedin:")
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
    option_type = st.sidebar.radio("Option Type", ['call', 'put'])

    # Calculate option price
    if st.sidebar.button("Calculate Option Price"):
        option_price = bsm_option_price(S, K, T, r, sigma, option_type)
        st.success(f"The theoretical price of the {option_type} option is: **${option_price:.2f}**")

    # Heatmap Section
    st.header("Interactive Heatmap: Option Price vs. Stock Price and Time to Maturity")
    stock_prices = np.linspace(0.5 * S, 1.5 * S, 50)
    times_to_maturity = np.linspace(0.1, T, 50)
    heatmap_data = []

    for s in stock_prices:
        for t in times_to_maturity:
            call_price = bsm_option_price(s, K, t, r, sigma, 'call')
            put_price = bsm_option_price(s, K, t, r, sigma, 'put')
            heatmap_data.append([s, t, call_price, put_price])

    heatmap_df = pd.DataFrame(heatmap_data, columns=["Stock Price", "Time to Maturity", "Call Price", "Put Price"])

    # Heatmap for Call Options
    heatmap_pivot_call = heatmap_df.pivot(index="Stock Price", columns="Time to Maturity", values="Call Price")
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_pivot_call, cmap="RdYlGn", annot=False, fmt="1.2f", cbar_kws={"label": "Call Price"})
    plt.title("Call Option Price Heatmap")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Stock Price")
    st.pyplot(plt)
    plt.clf()  # Clear the figure

    # Heatmap for Put Options
    heatmap_pivot_put = heatmap_df.pivot(index="Stock Price", columns="Time to Maturity", values="Put Price")
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_pivot_put, cmap="RdYlGn", annot=False, fmt="1.2f", cbar_kws={"label": "Put Price"})
    plt.title("Put Option Price Heatmap")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Stock Price")
    st.pyplot(plt)

    # Display formulas
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
