import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter


# Configurable parameters

NUM_RETRAINS = 10
ROLLING_WINDOW = 1100
START_DATE = '2019-06-01'      # Dates must be in YYYY-MM-DD format
END_DATE = '2025-06-01'


scaler = StandardScaler()

def bad_step(monitor):
    hist = getattr(monitor, "history", [])
    if len(hist) < 2:
        return False
    return (np.diff(hist) < 0).any()

NUM_STATES = 3 # Do not change

data = yf.download('^GSPC', start=START_DATE, end=END_DATE, progress=False)
data['log_rets'] = np.log(data['Close'] / data['Close'].shift(1))
data['range'] = (data['High'] / data['Low']) - 1
data = data.drop(columns=['Volume', 'Low', 'High']).dropna()


trades = []

for i in range(len(data) - (ROLLING_WINDOW + 3)):
    hidden_states_series = []
    for j in range(NUM_RETRAINS):
        model_train = data.iloc[i:ROLLING_WINDOW + i].copy()
        X = scaler.fit_transform(model_train[['log_rets', 'range']])
        while True:
            retrain = False

            hmm_model = GaussianHMM(n_components=NUM_STATES, covariance_type='diag', n_iter=100, min_covar=1e-9).fit(X)
            if bad_step(hmm_model.monitor_):
                retrain = True

            model_df = model_train.copy()
            hidden_states = hmm_model.predict(X)
            model_df['hidden_state'] = hidden_states

            state_0 = model_df[model_df['hidden_state'] == 0]
            state_1 = model_df[model_df['hidden_state'] == 1]
            state_2 = model_df[model_df['hidden_state'] == 2]

            counts = np.bincount(hidden_states, minlength=NUM_STATES)
            if (counts < 0.02 * len(model_train)).any():
                retrain = True

            if not retrain:
                break

        state_means = {0: float(state_0['log_rets'].mean()), 1: float(state_1['log_rets'].mean()), 2: float(state_2['log_rets'].mean())}
        sorted_means = dict(sorted(state_means.items(), key=lambda item: item[1]))
        rank = {state: idx for idx, state in enumerate(sorted_means)}
        model_df['Regime'] = model_df['hidden_state'].map(rank)
        hidden_states_series.append({f'{j}': model_df['Regime'].iloc[-1]})
    values = [list(d.values())[0] for d in hidden_states_series]
    counts = Counter(values)
    most_common_value = counts.most_common(1)[0][0]

    if most_common_value == 2:
        trades.append({'date': data.index[ROLLING_WINDOW + 1 + i], 'pct_ret': ((float(data['Open'].iloc[i + ROLLING_WINDOW + 2].iloc[0]) / float(data['Open'].iloc[ROLLING_WINDOW + 1 + i].iloc[0])) - 1), 'trade': 1})
    else:
        trades.append({'date': data.index[ROLLING_WINDOW + 1 + i], 'pct_ret': 0, 'trade': 0})

trades_df = pd.DataFrame(trades)
trades_df['cum_ret'] = (1 + trades_df['pct_ret']).cumprod() - 1
trades_df.set_index('date', inplace=True)
years = (len(trades_df) / 252)
cagr = ((1 + trades_df['cum_ret'].iloc[-1]) ** (1 / years)) - 1
mean = trades_df[trades_df['trade'] == 1]['pct_ret'].mean()
std = trades_df[trades_df['trade'] == 1]['pct_ret'].std(ddof=1)
sharpe = ((mean - (((1.05)**(1/252)) - 1)) / std) * np.sqrt(252)

sp500 = data[ROLLING_WINDOW:][['Close']]
sp500['rets'] = sp500['Close'].pct_change()
sp500['cum_ret'] = (1 + sp500['rets']).cumprod() - 1


plt.figure(figsize=(12, 6))
plt.title(f'S&P 500 Strategy Performance\nCAGR: {cagr:.2%} | Sharpe Ratio: {sharpe:.2f}')
plt.plot(trades_df['cum_ret'], label='Strategy returns', color='blue')
plt.plot(sp500['cum_ret'], label='Normal returns', color='orange')
plt.legend()
plt.show()

