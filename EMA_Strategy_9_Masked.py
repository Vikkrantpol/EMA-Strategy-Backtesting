import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm

# Set Seaborn style for clean, professional plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (14, 8)

# Load the CSV data
df = pd.read_csv('DOGEdata4h.csv')

# Print column names for debugging
print("Column names in CSV:", df.columns.tolist())

# Convert Unix time to datetime
df['time'] = pd.to_datetime(df['time'], unit='s')

# Sort by time to ensure chronological order
df = df.sort_values('time').reset_index(drop=True)

# Define exchange rate (USD to INR)
exchange_rate = 80  # Fixed rate for simplicity

# Convert price columns from USD to INR
for col in ['close', 'high', 'low', 'open']:
    df[col] *= exchange_rate

# [Proprietary Trading Logic Hidden]
# Note: The following section for calculating trading signals and executing trades
# is proprietary and has been omitted. The script assumes signals and trades are
# pre-computed and stored in df['long_signal'], df['short_signal'], and trades list.

# Initialize variables for backtesting
initial_capital = 10000000  # INR
realized_profit = 0
position = 0
entry_price = None
entry_time = None
equity_curve = []
drawdowns = []
trades = []

# Simulate the trading strategy (abstracted)
for i in range(1, len(df)):
    # Handle long signal
    if df.get('long_signal', pd.Series([False] * len(df)))[i-1]:
        if position < 0:  # Close short position
            exit_price = df['open'][i]
            profit = position * (exit_price - entry_price)
            capital_at_entry = initial_capital + realized_profit
            profit_percent = (profit / capital_at_entry) * 100 if capital_at_entry != 0 else 0
            exit_time = df['time'][i]
            duration_hours = (exit_time - entry_time).total_seconds() / 3600
            realized_profit += profit
            capital_after_trade = initial_capital + realized_profit
            trades.append({
                'type': 'short',
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'profit': profit,
                'profit_percent': profit_percent,
                'capital_after_trade': capital_after_trade,
                'duration_hours': duration_hours
            })
            position = 0
        # Open long position
        capital = initial_capital + realized_profit
        entry_price = df['open'][i]
        entry_time = df['time'][i]
        position = capital / entry_price
    
    # Handle short signal
    elif df.get('short_signal', pd.Series([False] * len(df)))[i-1]:
        if position > 0:  # Close long position
            exit_price = df['open'][i]
            profit = position * (exit_price - entry_price)
            capital_at_entry = initial_capital + realized_profit
            profit_percent = (profit / capital_at_entry) * 100 if capital_at_entry != 0 else 0
            exit_time = df['time'][i]
            duration_hours = (exit_time - entry_time).total_seconds() / 3600
            realized_profit += profit
            capital_after_trade = initial_capital + realized_profit
            trades.append({
                'type': 'long',
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'profit': profit,
                'profit_percent': profit_percent,
                'capital_after_trade': capital_after_trade,
                'duration_hours': duration_hours
            })
            position = 0
        # Open short position
        capital = initial_capital + realized_profit
        entry_price = df['open'][i]
        entry_time = df['time'][i]
        position = - (capital / entry_price)
    
    # Calculate equity and drawdown
    if position != 0:
        current_price = df['close'][i]
        unrealized_profit = position * (current_price - entry_price)
        equity = initial_capital + realized_profit + unrealized_profit
    else:
        equity = initial_capital + realized_profit
    equity_curve.append(equity)
    
    # Calculate running drawdown
    peak = max(equity_curve)
    drawdown = (peak - equity) / peak * 100
    drawdowns.append(drawdown)

# Close any remaining open position
if position != 0:
    exit_price = df['close'].iloc[-1]
    profit = position * (exit_price - entry_price)
    capital_at_entry = initial_capital + realized_profit
    profit_percent = (profit / capital_at_entry) * 100 if capital_at_entry != 0 else 0
    exit_time = df['time'].iloc[-1]
    duration_hours = (exit_time - entry_time).total_seconds() / 3600
    realized_profit += profit
    capital_after_trade = initial_capital + realized_profit
    trades.append({
        'type': 'long' if position > 0 else 'short',
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'profit': profit,
        'profit_percent': profit_percent,
        'capital_after_trade': capital_after_trade,
        'duration_hours': duration_hours
    })

# Save trades to CSV
trades_df = pd.DataFrame(trades)
trades_df.to_csv('trades_1h.csv', index=False)

# Calculate final equity and performance metrics
final_equity = initial_capital + realized_profit
percentage_return = (final_equity - initial_capital) / initial_capital * 100
max_drawdown = max(drawdowns) if drawdowns else 0
num_trades = len(trades)
win_trades = sum(1 for trade in trades if trade['profit'] > 0)
win_rate = win_trades / num_trades * 100 if num_trades > 0 else 0

# Calculate highest equity return
equity_percent_returns = [(equity - initial_capital) / initial_capital * 100 for equity in equity_curve]
max_equity_return_percent = max(equity_percent_returns) if equity_percent_returns else 0
max_equity_return_idx = np.argmax(equity_percent_returns) if equity_percent_returns else 0
max_equity_return_time = df['time'][1 + max_equity_return_idx] if equity_percent_returns else df['time'].iloc[-1]
max_equity_return_inr = equity_curve[max_equity_return_idx] - initial_capital if equity_percent_returns else 0

# Calculate monthly returns
equity_df = pd.DataFrame({'time': df['time'][1:], 'equity': equity_curve})
equity_df['month'] = equity_df['time'].dt.to_period('M')
monthly_equity = equity_df.groupby('month').agg({'equity': ['first', 'last']}).reset_index()
monthly_equity.columns = ['month', 'equity_start', 'equity_end']
monthly_equity['return_percent'] = ((monthly_equity['equity_end'] - monthly_equity['equity_start']) / monthly_equity['equity_start']) * 100
monthly_equity['month_str'] = monthly_equity['month'].apply(lambda x: x.strftime('%B %Y'))
monthly_returns = monthly_equity[['month', 'return_percent']].to_dict('records')
monthly_returns_str = [f"{month.month_str}: {ret:.2f}%" for month, ret in zip(monthly_equity.itertuples(), monthly_equity['return_percent'])]

# Calculate monthly number of trades
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
monthly_trades = trades_df.groupby('month').size().reset_index(name='num_trades')
monthly_trades['month_str'] = monthly_trades['month'].apply(lambda x: x.strftime('%B %Y'))

# Find best and worst trades
best_trade = max(trades, key=lambda x: x['profit'], default={'profit': 0, 'exit_time': df['time'].iloc[-1]})
worst_trade = min(trades, key=lambda x: x['profit'], default={'profit': 0, 'exit_time': df['time'].iloc[-1]})

# Print results
print(f"Initial Capital: {initial_capital:,} INR")
print(f"Final Equity: {final_equity:,.2f} INR")
print(f"Percentage Return: {percentage_return:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Number of Trades: {num_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Best Trade Profit: {best_trade['profit']:,.2f} INR at {best_trade['exit_time']}")
print(f"Worst Trade Profit: {worst_trade['profit']:,.2f} INR at {worst_trade['exit_time']}")
print(f"Highest Equity Return: {max_equity_return_inr:,.2f} INR ({max_equity_return_percent:.2f}%) at {max_equity_return_time}")
print("Monthly Returns:")
for mr in monthly_returns_str:
    print(mr)

# --- Visualizations ---

# 1. Equity Curve with DOGE Price
fig, ax1 = plt.subplots()
ax1.plot(df['time'][1:], equity_curve, color='blue', label='Equity (INR)')
ax1.fill_between(df['time'][1:], equity_curve, initial_capital, where=np.array(equity_curve) >= initial_capital, facecolor='green', alpha=0.2)
ax1.fill_between(df['time'][1:], equity_curve, initial_capital, where=np.array(equity_curve) < initial_capital, facecolor='red', alpha=0.2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Equity (INR)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)

# Secondary axis for DOGE price
ax2 = ax1.twinx()
ax2.plot(df['time'][1:], df['close'][1:], color='orange', alpha=0.5, label='DOGE Price (INR)')
ax2.set_ylabel('DOGE Price (INR)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Annotate peak equity, max drawdown, and highest equity return
peak_idx = np.argmax(equity_curve)
ax1.annotate(f'Peak: {equity_curve[peak_idx]:,.0f} INR', xy=(df['time'][1+peak_idx], equity_curve[peak_idx]), 
             xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->'), color='blue')
drawdown_idx = np.argmax(drawdowns)
ax1.annotate(f'Max DD: {max_drawdown:.2f}%', xy=(df['time'][1+drawdown_idx], equity_curve[drawdown_idx]), 
             xytext=(10, -20), textcoords='offset points', arrowprops=dict(arrowstyle='->'), color='red')
ax1.annotate(f'Max Return: {max_equity_return_inr:,.0f} INR ({max_equity_return_percent:.2f}%)', 
             xy=(df['time'][1+max_equity_return_idx], equity_curve[max_equity_return_idx]), 
             xytext=(10, 30), textcoords='offset points', arrowprops=dict(arrowstyle='->'), color='green')

plt.title('Equity Curve vs DOGE Price')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.savefig('equity_curve_1h.png', dpi=600)
plt.close()

# 2. Drawdown Plot
plt.figure()
plt.plot(df['time'][1:], drawdowns, color='red', label='Drawdown (%)')
plt.fill_between(df['time'][1:], drawdowns, 0, where=np.array(drawdowns) > 10, facecolor='red', alpha=0.3)
plt.axhline(y=max_drawdown, color='black', linestyle='--', alpha=0.5)
plt.annotate(f'Max DD: {max_drawdown:.2f}% at {df['time'][1+drawdown_idx]}', xy=(df['time'][1+drawdown_idx], max_drawdown), 
             xytext=(10, 10), textcoords='offset points', color='black')
plt.xlabel('Time')
plt.ylabel('Drawdown (%)')
plt.title('Drawdown Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('drawdown_1h.png', dpi=600)
plt.close()

# 3. Price Chart with Indicators
plt.figure()
plt.plot(df['time'], df['close'], color='black', alpha=0.5, label='DOGE Price (INR)')
# [Proprietary Indicators Hidden]
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Price (INR, Log Scale)')
plt.title('DOGE Price with Indicators')
plt.legend()
plt.tight_layout()
plt.savefig('price_signals_1h.png', dpi=600)
plt.show()

# 4. Monthly Returns and Trades Plot
fig, ax1 = plt.subplots(figsize=(16, 8))
norm = plt.Normalize(min(monthly_equity['return_percent']), max(monthly_equity['return_percent']))
colors = cm.viridis(norm(monthly_equity['return_percent']))
bars = ax1.bar(monthly_equity['month_str'], monthly_equity['return_percent'], color=colors, label='Monthly Return (%)', edgecolor='black')
ax1.set_xlabel('Month', fontsize=14)
ax1.set_ylabel('Return (%)', color='navy', fontsize=14)
ax1.tick_params(axis='y', labelcolor='navy')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, linestyle='--')

# Annotate highest and lowest returns
max_return_idx = monthly_equity['return_percent'].idxmax()
min_return_idx = monthly_equity['return_percent'].idxmin()
ax1.annotate(f'Highest: {monthly_equity["return_percent"][max_return_idx]:.2f}%', 
             xy=(max_return_idx, monthly_equity['return_percent'][max_return_idx]), 
             xytext=(0, 10), textcoords='offset points', ha='center', color='green',
             arrowprops=dict(arrowstyle='->', color='green'))
ax1.annotate(f'Lowest: {monthly_equity["return_percent"][min_return_idx]:.2f}%', 
             xy=(min_return_idx, monthly_equity['return_percent'][min_return_idx]), 
             xytext=(0, -15), textcoords='offset points', ha='center', color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

# Secondary axis for number of trades
ax2 = ax1.twinx()
ax2.plot(monthly_trades['month_str'], monthly_trades['num_trades'], color='darkorange', marker='o', linewidth=2, markersize=8, label='Number of Trades')
ax2.set_ylabel('Number of Trades', color='darkorange', fontsize=14)
ax2.tick_params(axis='y', labelcolor='darkorange')

plt.title('Monthly Returns and Number of Trades', fontsize=16, pad=20)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)
plt.tight_layout()
plt.savefig('monthly_returns_trades.png', dpi=600)
plt.close()

# 5. Summary Dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

# Equity Curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['time'][1:], equity_curve, color='blue', label='Equity (INR)')
ax1.fill_between(df['time'][1:], equity_curve, initial_capital, where=np.array(equity_curve) >= initial_capital, facecolor='green', alpha=0.2)
ax1.fill_between(df['time'][1:], equity_curve, initial_capital, where=np.array(equity_curve) < initial_capital, facecolor='red', alpha=0.2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Equity (INR)')
ax1.set_title('Equity Curve')
ax1.grid(True, alpha=0.3)

# Drawdown
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(df['time'][1:], drawdowns, color='red', label='Drawdown (%)')
ax2.fill_between(df['time'][1:], drawdowns, 0, where=np.array(drawdowns) > 10, facecolor='red', alpha=0.3)
ax2.axhline(y=max_drawdown, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time')
ax2.set_ylabel('Drawdown (%)')
ax2.set_title('Drawdown')
ax2.legend()

# Metrics Table
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off')
metrics = [
    f"Initial Capital: {initial_capital:,} INR",
    f"Final Equity: {final_equity:,.2f} INR",
    f"Percentage Return: {percentage_return:.2f}%",
    f"Max Drawdown: {max_drawdown:.2f}% at {df['time'][1+drawdown_idx]}",
    f"Number of Trades: {num_trades}",
    f"Win Rate: {win_rate:.2f}%",
    f"Best Trade: {best_trade['profit']:,.2f} INR at {best_trade['exit_time']}",
    f"Worst Trade: {worst_trade['profit']:,.2f} INR at {worst_trade['exit_time']}",
    f"Highest Equity Return: {max_equity_return_inr:,.2f} INR ({max_equity_return_percent:.2f}%) at {max_equity_return_time}"
]
ax3.table(cellText=[[m] for m in metrics], loc='center', cellLoc='left', edges='open')
ax3.set_title('Performance Metrics')

plt.suptitle('Strategy Summary', fontsize=16)
plt.tight_layout()
plt.savefig('summary_dashboard_4h.png', dpi=600)
plt.close()
