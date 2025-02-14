import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm

# Black-Scholes-Merton Formula
def bsm_option_price(S, K, T, r, sigma, option_type):
    """
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

# Streamlit App
def main():
    st.title("Black-Scholes-Merton Option Pricing Model")
    st.write("This app calculates the theoretical price of European call and put options using the Black-Scholes-Merton formula.")

    # Input parameters
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
        st.success(f"The theoretical price of the {option_type} option is: **{option_price:.2f}**")

    # Heatmap Section
    st.header("Interactive Heatmap: Option Price vs. Stock Price and Time to Maturity")
    st.write("Explore how the option price varies with changes in stock price and time to maturity.")

    # Generate data for heatmap
    stock_prices = np.linspace(0.5 * S, 1.5 * S, 50)  # Range of stock prices (50% to 150% of S)
    times_to_maturity = np.linspace(1, 100.0, 5)     
    heatmap_data = []

    for s in stock_prices:
        for t in times_to_maturity:
            price = bsm_option_price(s, K, t, r, sigma, option_type)
            heatmap_data.append([s, t, price])

    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, columns=["Stock Price", "Time to Maturity", "Option Price"])

    # Reshape data for heatmap
    heatmap_pivot = heatmap_df.pivot(index="Stock Price", columns="Time to Maturity", values="Option Price")

    # Create interactive heatmap using Plotly
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Time to Maturity (Years)", y="Stock Price", color="Option Price"),
        x=times_to_maturity,
        y=stock_prices,
        color_continuous_scale="Viridis",
        title=f"{option_type.capitalize()} Option Price Heatmap"
    )
    fig.update_layout(
        xaxis_title="Time to Maturity (Years)",
        yaxis_title="Stock Price",
        coloraxis_colorbar=dict(title="Option Price")
    )
    st.plotly_chart(fig, use_container_width=True)

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