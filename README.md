# market_regime
A financial modeling project that uses a Gaussian Hidden Markov Model (HMM) to identify and classify market regimes.

## Overview
This project applies a rolling Gaussian HMM to historical market data in order to detect regime changes.
It uses a rolling observation window between a user-specified start and end date.
It classifies market conditions into three distinct regimes.
Overall it is a simple academic backtest that places trades based on the previous day’s detected regime.

In previous parameter testing, a window of roughly 1100 days has been found to yield the highest results. 
This is purely an academic backtest that does not account for market friction like fees or tax. 
In my experience it will not beat the market in normal returns, but risk adjusted returns tend to be better.
In the future I would like to expand upon the idea of this market regime model and possibly integrate it with other strategies rather than using it on its own.

## Requirements:

-Python 3.11

-Libraries: pandas, numpy, matplotlib, yfinance, sklearn, hmmlearn

## Usage:
-Clone repository

-pip install -r requirements.txt

-set parameters (START_DATE, END_DATE, ROLLING_WINDOW)

-run the model using  
python market_regime.py




