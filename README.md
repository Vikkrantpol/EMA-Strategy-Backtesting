# EMA-Strategy-Backtesting (DOGE Coin)

## Overview
This repository contains a Python script for backtesting a proprietary trading strategy on DOGE/USD data, converted to INR. The script processes hourly OHLC data, executes trades based on a hidden trading logic, and generates comprehensive performance visualizations and metrics. The trading logic is proprietary and not disclosed in the script.
Using 4h timeframe (16 months of historical data)
### Note: The strategy and its implementation are protected by an "All Rights Reserved" license. Any use, replication, or adaptation of the strategy for any purpose is strictly prohibited.

## Results

 The backtest yields the following performance metrics:
- Initial Capital: 10,000,000 INR
- Final Equity: 51,819,389.39 INR
- Percentage Return: 418.19%
- Max Drawdown: 38.25%
- Number of Trades: 102
- Win Rate: 35.29%
- Best Trade: 23,410,847.77 INR at 2024-11-17 08:00:00
- Worst Trade: -6,038,769.27 INR at 2025-03-04 04:00:00
- Highest Equity Return: 58,223,238.67 INR (582.23%) at 2025-03-02 20:00:00

 Monthly Returns:

- April 2024: 21.95%
- May 2024: -6.55%
- June 2024: 6.66%
- July 2024: 5.07%
- August 2024: 3.33%
- September 2024: 23.74%
- October 2024: 23.37%
- November 2024: 109.20%
- December 2024: -0.48%
- January 2025: 19.89%
- February 2025: 27.31%
- March 2025: -12.99%
- April 2025: -7.46%
- May 2025: 6.45%

## Plots and Visualizations

The following plots illustrate the strategy's performance:

### 1.Equity Curve vs. DOGE Price
- Shows equity growth from 10M INR to over 51.8M INR, compared to DOGE price, with gains (green) and losses (red) highlighted.
  

### 2.Drawdown Over Time
- Displays drawdown percentage, peaking at 38.25%, with significant drawdowns annotated.

### 3.Price Chart with Indicators
- Plots DOGE price with proprietary indicators (details hidden) on a logarithmic scale.

### 4.Monthly Returns and Number of Trades
- Combines monthly returns (bar plot with gradient colors, e.g., 109.20% in Nov 2024) and number of trades (line plot), with highest and lowest returns annotated.

### 5.Summary Dashboard
 -Aggregates equity curve, drawdown, and a metrics table for a comprehensive overview.

## Features
- Data Processing: Loads and processes hourly DOGE/USD data, converting prices to INR.

- Trade Simulation: Executes trades based on proprietary signals (not disclosed).

- Performance Metrics: Calculates final equity, percentage return, max drawdown, win rate, best/worst trades, and highest equity return.

### Visualizations:

- Equity curve vs. DOGE price

- Drawdown over time

- Price chart with indicators (details hidden)

- Monthly returns and number of trades

- Summary dashboard

### Output: Saves trade details to trades_1h.csv and visualizations as high-resolution PNG files.

## License

This project is licensed under an "All Rights Reserved" license. See the LICENSE file for details. Using, copying, or adapting the strategy for any purpose is strictly prohibited.

## Disclaimer

This script is for demonstration purposes only. The trading logic is proprietary and not fully disclosed. No guarantee of performance is provided. Use at your own risk, subject to the license terms.
