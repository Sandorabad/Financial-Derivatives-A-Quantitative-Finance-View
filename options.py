""" Contains functions and classes for options"""

"""
Copyright (c) 2019 Cameron R. Connell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from math import exp, log, sqrt
from scipy.stats import norm

HASNUMPY = 1
try:
    import numpy as np
except ImportError:
    print("Plotting functions require Numpy")
    print("Plotting functions inoperable")
    HASNUMPY = 0

HASMATPLOTLIB = 1
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Plotting functions require Matplotlib")
    print("Plotting functions inoperable")
    HASMATPLOTLIB = 0


def BSCall(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes call price.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes call price.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    d2 = (log(spot/strike)+(rate-vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return spot*norm.cdf(d1)-strike * exp(-rate*(expiry - time))*norm.cdf(d2)

def BSPut(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes put price.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes put price.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    d2 = (log(spot/strike)+(rate-vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return -spot*norm.cdf(-d1) + strike * exp(-rate*(expiry - time))*norm.cdf(-d2)

def BSCall_Delta(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes call delta.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes call delta.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return norm.cdf(d1)

def BSPut_Delta(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes put delta.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes put delta.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return -norm.cdf(-d1)

def BSCall_Gamma(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes call gamma.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes call gamma.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return norm.pdf(d1)/spot/vol/sqrt(expiry-time)

def BSPut_Gamma(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes put gamma.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes put gamma.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return norm.pdf(d1)/spot/vol/sqrt(expiry-time)

def BSCall_Theta(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes call theta, scaled to the 1 day change in option
        value due to time decay.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The 1 day Black-Scholes call theta.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    d2 = (log(spot/strike)+(rate-vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    theta = -spot*vol*norm.pdf(d1)/2/sqrt(expiry - time) - rate * strike *exp(-rate *(expiry-time)) * norm.cdf(d2)
    return theta / 365

def BSPut_Theta(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes put theta, scaled to the 1 day change in option
        value due to time decay.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The 1 day Black-Scholes put theta.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    d2 = (log(spot/strike)+(rate-vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    theta = -spot*vol*norm.pdf(d1)/2/sqrt(expiry - time) + rate * strike *exp(-rate *(expiry-time)) * norm.cdf(-d2)
    return theta / 365

def BSCall_Vega(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes call vega.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes call vega.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return spot*sqrt(expiry - time) * norm.pdf(d1) / 100

def BSPut_Vega(spot, time, strike, expiry, vol, rate):
    """
        Calculates the Black-Scholes call vega.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The time when the call price is to be evaluated.
        strike: float
            The strike price of the call.
        expiry: float
            The expiration date of the call.
        vol: float
            The implied volatility to use to price the call (as a percentage).
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes call vega.
    """

    vol /= 100
    rate /= 100
    d1 = (log(spot/strike)+(rate+vol**2/2) * (expiry - time)) / vol / sqrt(expiry - time)
    return spot*sqrt(expiry - time) * norm.pdf(d1) / 100


def implied_vol(price, spot, strike, expiry, rate):
    """
        This function provides a partial implementation of Jaeckel's optimized algorithm
        for computing implied volatility (https://jaeckel.000webhostapp.com/ByImplication.pdf)
        Currently, only calls are implemented.

        Parameters
        ----------
        price: float
            The option price.
        spot: float
            The spot price of the underlying.
        strike: float
            The strike price of the call.
        expiry: float
            The time to expiration.
        rate: float
            The risk free interest rate to use in the model (as a percentage).

        Returns
        -------
        float
            The Black-Scholes implied volatility corresponding to the entered price.
    """


    rate /= 100
    tolerance = 0.00000001

    theta = 1

    if spot - exp(-rate * expiry) * strike >= price or price >= spot:
        print("Option price out of range")
        return
    

    x=log(exp(rate*expiry)*spot/strike)
    scaled_price = price * exp(rate*expiry/2) / sqrt(spot*strike)

    def F(sigma):
        return theta * exp(x/2) * norm.cdf(theta*(x/sigma + sigma/2)) \
            - theta * exp(-x/2) * norm.cdf(theta*(x/sigma - sigma/2))
    
    def Fprime(sigma):
        return exp(x/2)*norm.pdf(theta*(x/sigma + sigma/2)) * (-x/sigma**2 + 0.5) \
            - exp(-x/2) * norm.pdf(theta * (x/sigma - sigma/2)) * (-x/sigma**2 - 0.5)

    sigma_c = sqrt(2 * abs(x))

    b_c = F(sigma_c)

    if scaled_price >= b_c:
        pval = (exp(theta*x/2)-scaled_price) * norm.cdf(-sqrt(abs(x)/2))/(exp(theta*x/2)-b_c)
        old_sigma = -2 * norm.ppf(pval)
        new_sigma = old_sigma - (F(old_sigma) - scaled_price)/Fprime(old_sigma)

        while abs(new_sigma - old_sigma) > tolerance:
            old_sigma = new_sigma
            new_sigma = old_sigma - (F(old_sigma) - scaled_price)/Fprime(old_sigma)

        return 100*new_sigma/sqrt(expiry)
    else:
        if theta * x <= 0:
            iota = 0
        else:
            iota = theta * (exp(x/2) - exp(-x/2))
        
        def G(sigma):
            return log(F(sigma) - iota) - log(scaled_price - iota)
        
        def Gprime(sigma):
            return Fprime(sigma)/(F(sigma) - iota)

        old_sigma = sqrt(2*x**2/(abs(x)-4 * log((scaled_price - iota)/(b_c - iota))))
        new_sigma = old_sigma - G(old_sigma)/Gprime(old_sigma)

        while abs(new_sigma - old_sigma) > tolerance:
            old_sigma = new_sigma
            new_sigma = old_sigma - G(old_sigma)/Gprime(old_sigma)

        return 100*new_sigma/sqrt(expiry)


class option:
    """
    Class for option products.

    Attributes
    ----------
    strike: float
        The strike price of the option.
    expiry: float
        The expiration date of the option, in years.
    type: string
        Either "call" or "put" indicating the option type.
    """


    def __init__(self, strike=0.0, expiry=0.0, type="call"):
        self.strike = strike
        self.expiry = expiry
        self.type = type

    def get_strike(self):
        """
        Returns the value of the strike attribute, the strike price of the option.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The strike price.
        """

        return self.strike
    
    def get_expiry(self):
        """
        Returns the value of the expiry attribute, the expiration date of the option

        Parameters
        ----------
        None

        Returns
        -------
        float
            The expiration date.
        """
        return self.expiry
    
    def get_type(self):
        """
        Returns the option type.

        Parameters
        ----------
        None

        Returns
        -------
        string
            The option type.
        """
        return self.type

    def set_strike(self, strike=None):
        """
        Sets the strike price of the option, overwriting any existing value for the
        strike attribute.

        Parameters
        ----------
        strike: float
            The new strike.

        Returns
        -------
        None
        """
        if strike is None:
            print("Must provide a strike")
            return None
        
        self.strike = strike
    
    def set_expiry(self, expiry):
        """
        Sets the expiration date for the option, overwriting any existing value for the
        expiry attribute.

        Parameters
        ----------
        expiry: float
            The new expiration date (in years).

        Returns
        -------
        None
        """
        if expiry is None:
            print("Must provide an expiry")
            return None
        
        self.expiry = expiry

    def price(self, spot, time, vol, rate):
        """
        Returns the option price.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        float
            The option price or premium.
        """
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None
        
        if self.type == "call":
            return BSCall(spot, time, self.strike, self.expiry, vol, rate)
        else:
            return BSPut(spot, time, self.strike, self.expiry, vol, rate)

    
    def delta(self, spot, time, vol, rate):
        """
        Returns the option delta.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        float
            The option delta.
        """
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None

        if self.type == "call":
            return BSCall_Delta(spot, time, self.strike, self.expiry, vol, rate)
        else:
            return BSPut_Delta(spot, time, self.strike, self.expiry, vol, rate)

    
    def gamma(self, spot, time, vol, rate):
        """
        Returns the option gamma.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        float
            The option gamma.
        """
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None

        if self.type == "call":
            return BSCall_Gamma(spot, time, self.strike, self.expiry, vol, rate)
        else:
            return BSPut_Gamma(spot, time, self.strike, self.expiry, vol, rate)

    
    def vega(self, spot, time, vol, rate):
        """
        Returns the option vega.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        float
            The option vega.
        """
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None

        if self.type == "call":
            return BSCall_Vega(spot, time, self.strike, self.expiry, vol, rate)
        else:
            return BSPut_Vega(spot, time, self.strike, self.expiry, vol, rate)

    
    def theta(self, spot, time, vol, rate):
        """
        Returns the option theta.

        Parameters
        ----------
        spot: float
            The spot price of the underlying.
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        float
            The option theta.
        """
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None

        if self.type == "call":
            return BSCall_Theta(spot, time, self.strike, self.expiry, vol, rate)
        else:
            return BSPut_Theta(spot, time, self.strike, self.expiry, vol, rate)
    

    def plot_payoff(self):
        """
        Plots the payoff (at expiration) of the option.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not HASMATPLOTLIB:
            print("Plotting functions require Matplotlib")
            return None
        
        fig, ax = plt.subplots()


        s=np.arange(0.0, 2*self.strike, 0.1)

        p=[]
        for spot in s:
            if self.type == "call":
                price = max(0, spot - self.strike)
            else:
                price = max(0, self.strike - spot)

            p.append(price)
        
        payoff = np.array(p)

        ax.plot(s, payoff)

        ax.set(xlabel='spot', ylabel='payoff',
            title = 'Option Payoff')
        
        plt.show(block=False)


    def plot_price(self, time, vol, rate):
        """
        Plots the option price, on the given date, against the underlying spot price.

        Parameters
        ----------
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        None
        """
        if not HASMATPLOTLIB:
            print("Plotting functions require Matplotlib")
            return None
        
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None
        
        fig, ax = plt.subplots()


        s=np.arange(0.1, 2*self.strike, 0.1)

        p=[]
        for spot in s:
            if self.type == "call":
                price = BSCall(spot, time, self.strike, self.expiry, vol, rate)
            else:
                price = BSPut(spot, time, self.strike, self.expiry, vol, rate)

            p.append(price)
        
        prices = np.array(p)

        ax.plot(s, prices)

        ax.set(xlabel='spot', ylabel='price',
            title = 'Option Price vs. Spot Price')
        
        plt.show(block=False)
    
    
    def plot_delta(self, time, vol, rate):
        """
        Plots the option delta against the spot price of the underlying.

        Parameters
        ----------
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        None
        """
        if not HASMATPLOTLIB:
            print("Plotting functions require Matplotlib")
            return None
        
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None
        
        fig, ax = plt.subplots()


        s=np.arange(0.1, 2*self.strike, 0.1)

        d=[]
        for spot in s:
            if self.type == "call":
                delta = BSCall_Delta(spot, time, self.strike, self.expiry, vol, rate)
            else:
                delta = BSPut_Delta(spot, time, self.strike, self.expiry, vol, rate)

            d.append(delta)
        
        deltas = np.array(d)

        ax.plot(s, deltas)

        ax.set(xlabel='spot', ylabel='delta',
            title = 'Option Delta vs. Spot Price')
        
        plt.show(block=False)
    
    def plot_gamma(self, time, vol, rate):
        """
        Plots the option gamma against the spot price of the underlying.

        Parameters
        ----------
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        None
        """
        if not HASMATPLOTLIB:
            print("Plotting functions require Matplotlib")
            return None
        
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None
        
        fig, ax = plt.subplots()


        s=np.arange(0.1, 2*self.strike, 0.1)

        g=[]
        for spot in s:
            if self.type == "call":
                gamma = BSCall_Gamma(spot, time, self.strike, self.expiry, vol, rate)
            else:
                gamma = BSPut_Gamma(spot, time, self.strike, self.expiry, vol, rate)

            g.append(gamma)
        
        gammas = np.array(g)

        ax.plot(s, gammas)

        ax.set(xlabel='spot', ylabel='gamma',
            title = 'Option Gamma vs. Spot Price')
        
        plt.show(block=False)

    
    def plot_vega(self, time, vol, rate):
        """
        Plots the option vega against the spot price of the underlying.

        Parameters
        ----------
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        None
        """
        if not HASMATPLOTLIB:
            print("Plotting functions require Matplotlib")
            return None
        
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None
        
        fig, ax = plt.subplots()


        s=np.arange(0.1, 2*self.strike, 0.1)

        v=[]
        for spot in s:
            if self.type == "call":
                vega = BSCall_Vega(spot, time, self.strike, self.expiry, vol, rate)
            else:
                vega = BSPut_Vega(spot, time, self.strike, self.expiry, vol, rate)

            v.append(vega)
        
        vegas = np.array(v)

        ax.plot(s, vegas)

        ax.set(xlabel='spot', ylabel='vega',
            title = 'Option Vega vs. Spot Price')
        
        plt.show(block=False)

    
    def plot_theta(self, time, vol, rate):
        """
        Plots the theta of the option against the spot price of the underlying.

        Parameters
        ----------
        time: float
            The date the option should be priced for.
        vol: float
            The implied volatility to use for pricing.
        rate: float
            The risk free interest rate to use (as a percantage).

        Returns
        -------
        None
        """
        if not HASMATPLOTLIB:
            print("Plotting functions require Matplotlib")
            return None
        
        if time > self.expiry:
            print("Evaluation time must precede expiry")
            return None
        
        fig, ax = plt.subplots()


        s=np.arange(0.1, 2*self.strike, 0.1)

        t=[]
        for spot in s:
            if self.type == "call":
                theta = BSCall_Theta(spot, time, self.strike, self.expiry, vol, rate)
            else:
                theta = BSPut_Theta(spot, time, self.strike, self.expiry, vol, rate)

            t.append(theta)
        
        thetas = np.array(t)

        ax.plot(s, thetas)

        ax.set(xlabel='spot', ylabel='theta',
            title = 'Option Theta vs. Spot Price')
        
        plt.show(block=False)



    
