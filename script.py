import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

RISK_FREE = 0.05

def bs_call(S,K,T,r,s):
    d1=(np.log(S/K)+(r+0.5*s*s)*T)/(s*np.sqrt(T))
    d2=d1-s*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(price,S,K,T,r):
    intrinsic=max(S-K*np.exp(-r*T),0)
    if price<=intrinsic or T<=0:
        return np.nan
    f=lambda sigma: bs_call(S,K,T,r,sigma)-price
    try:
        return brentq(f,1e-4,5.0)
    except ValueError:
        return np.nan

def choose_expiry(expiries):
    today=datetime.now()
    best=None
    for e in expiries:
        d=datetime.strptime(e,"%Y-%m-%d")
        days=(d-today).days
        if 30<=days<=60:
            return e
        if days>30 and best is None:
            best=e
    return best or expiries[-1]

def build_smile(ticker):
    tk=yf.Ticker(ticker)
    S=tk.history(period="1d")["Close"].iloc[-1]
    expiry=choose_expiry(tk.options)
    calls=tk.option_chain(expiry).calls
    T=(datetime.strptime(expiry,"%Y-%m-%d")-datetime.now()).total_seconds()/(365.25*24*3600)
    moneyness=[]
    ivs=[]
    for _,row in calls.iterrows():
        bid=row["bid"]; ask=row["ask"]; last=row["lastPrice"]; K=row["strike"]
        if bid>0 and ask>0:
            price=0.5*(bid+ask)
        elif last>0:
            price=last
        else:
            continue
        iv=implied_vol(price,S,K,T,RISK_FREE)
        if np.isnan(iv) or iv>2:
            continue
        moneyness.append(K/S)
        ivs.append(iv*100)
    idx=np.argsort(moneyness)
    return np.array(moneyness)[idx],np.array(ivs)[idx],expiry

mA,ivA,eA=build_smile("AAPL")
mM,ivM,eM=build_smile("MSFT")

plt.figure(figsize=(9,6))
plt.plot(mA,ivA,'o-',label=f"AAPL ({eA})")
plt.plot(mM,ivM,'s-',label=f"MSFT ({eM})")
plt.xlabel("Moneyness (K/S)")
plt.ylabel("Implied Volatility (%)")
plt.title("Implied Volatility Smile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
