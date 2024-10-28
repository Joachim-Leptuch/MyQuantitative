import math
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.stats import norm

# Modelling

def terminal_stock_price_GBM(S0, r, sigma, t, N):

    t /= 365
    r /= 100
    sigma /= 100

    # Calculate the terminal stock price using the Geometric Brownian Motion equation
    ST = S0 * math.exp((r - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * N)

    return ST

# Volatility estimation

def closeToCloseVolatility(df_OHLC, nPeriods, periodsPerYear):
    
    df_OHLC["Returns"] = np.log(df_OHLC["Close"] / df_OHLC["Close"].shift(1))
    closeToCloseVolatility = 19.896 * df_OHLC["Returns"].abs().rolling(window = nPeriods).sum() / nPeriods
    #variance = (df_OHLC["Returns"] ** 2).rolling(window = nPeriods).sum() / nPeriods
    #closeToCloseVolatility = np.sqrt(variance) * np.sqrt(periodsPerYear)
    closeToCloseVolatility.dropna()
    closeToCloseVolatility.name = "Close to Close Ann. σ"
    
    return closeToCloseVolatility

def parkinsonVolatility(df_OHLC, nPeriods, periodsPerYear):
    
    biais = {
        '5':0.55,
        '10': 0.65,
        '20': 0.74,
        '50': 0.82,
        '100': 0.86,
        '200':0.92}
    
    df_OHLC["HighLow"] = np.log(df_OHLC["High"] / df_OHLC["Low"])
    parkinsonVolatility = np.sqrt((1 / (4 * nPeriods * np.log(2))) * ((df_OHLC['HighLow']**2).rolling(window=nPeriods).sum()))
    parkinsonVolatility /= biais[str(nPeriods)]
    parkinsonVolatility *= np.sqrt(periodsPerYear)
    parkinsonVolatility.name = "Parkinson Ann. σ"
    
    return parkinsonVolatility

def garhmanKlassVolatility(df_OHLC, nPeriods, periodsPerYear):
    
    biais = {
        '5':0.38,
        '10': 0.51,
        '20': 0.64,
        '50': 0.73,
        '100': 0.80,
        '200':0.85}
    
    df_OHLC["HighLow"] = np.log(df_OHLC["High"] / df_OHLC["Low"])
    df_OHLC["Returns"] = np.log(df_OHLC["Close"] / df_OHLC["Close"].shift(1))
    garhmanKlassVolatility = (((df_OHLC["HighLow"] ** 2) / 2).rolling(window = nPeriods).sum()) / nPeriods - (((2 * np.log(2) - 1) * (df_OHLC["Returns"] ** 2)).rolling(nPeriods).sum() / nPeriods)
    garhmanKlassVolatility /= biais[str(nPeriods)]
    garhmanKlassVolatility = np.sqrt(garhmanKlassVolatility) * np.sqrt(periodsPerYear)
    garhmanKlassVolatility.name = "Garhman-Klass Ann. σ"
    
    return garhmanKlassVolatility

def rogersSatchellVolatility(df_OHLC, nPeriods, periodsPerYear):
    
    df_OHLC["HighClose"] = np.log(df_OHLC["High"] / df_OHLC["Close"])
    df_OHLC["HighOpen"] = np.log(df_OHLC["High"] / df_OHLC["Open"])
    df_OHLC["LowClose"] = np.log(df_OHLC["Low"] / df_OHLC["Close"])
    df_OHLC["LowOpen"] = np.log(df_OHLC["Low"] / df_OHLC["Open"])
    rogersSatchellVolatility = ((df_OHLC["HighClose"] * df_OHLC["HighOpen"] + df_OHLC["LowClose"] * df_OHLC["LowOpen"]) / nPeriods).rolling(window = nPeriods).sum()
    rogersSatchellVolatility = np.sqrt(rogersSatchellVolatility) * np.sqrt(periodsPerYear)
    rogersSatchellVolatility.name = "Rogers-Satchell Ann. σ"
    
    return rogersSatchellVolatility

def yangZhangVolatility(df_OHLC, nPeriods, periodsPerYear):
    
    df_OHLC["HighClose"] = np.log(df_OHLC["High"] / df_OHLC["Close"])
    df_OHLC["HighOpen"] = np.log(df_OHLC["High"] / df_OHLC["Open"])
    df_OHLC["LowClose"] = np.log(df_OHLC["Low"] / df_OHLC["Close"])
    df_OHLC["LowOpen"] = np.log(df_OHLC["Low"] / df_OHLC["Open"])
    df_OHLC["OpenReturns"] = np.log(df_OHLC["Open"] / df_OHLC["Open"].shift(1))
    df_OHLC["CloseReturns"] = np.log(df_OHLC["Close"] / df_OHLC["Close"].shift(1))
    k = 0.34 / (1.34 + (nPeriods + 1) / (nPeriods - 1))
    varO = ((df_OHLC["OpenReturns"] - (df_OHLC["OpenReturns"].rolling(nPeriods).sum() / nPeriods)) ** 2).rolling(nPeriods).sum() / (nPeriods - 1)
    varC = ((df_OHLC["CloseReturns"] - (df_OHLC["CloseReturns"].rolling(nPeriods).sum() / nPeriods)) ** 2).rolling(nPeriods).sum() / (nPeriods - 1)
    varRC = ((df_OHLC["HighClose"] * df_OHLC["HighOpen"] + df_OHLC["LowClose"] * df_OHLC["LowOpen"]) / nPeriods).rolling(window = nPeriods).sum()
    yangZhangVolatility = varO + k * varC + (1 - k) * varRC
    yangZhangVolatility =  np.sqrt(yangZhangVolatility) * np.sqrt(periodsPerYear)
    yangZhangVolatility.name = "Yang-Zhang Ann. σ"
    
    return yangZhangVolatility

def instantVolatilityEstimation(df_OHLC, nPeriods, periodsPerYear):
    """
    Returns a dataframe with estimated volatilities for actual period.
    """
    
    volatilityEstimators = pd.DataFrame.from_dict({
    
    "CloseToClose" : closeToCloseVolatility(df_OHLC, nPeriods, periodsPerYear).iloc[-1], 
    "Parkinson" : parkinsonVolatility(df_OHLC, nPeriods, periodsPerYear).iloc[-1], 
    "Garman-Klass": garhmanKlassVolatility(df_OHLC, nPeriods, periodsPerYear).iloc[-1], 
    "Rogers-Satchell" : rogersSatchellVolatility(df_OHLC, nPeriods, periodsPerYear).iloc[-1], 
    "Yang-Zhang" : yangZhangVolatility(df_OHLC, nPeriods, periodsPerYear).iloc[-1]
    }, orient='index', columns=["Ann. Volatility"])
    
    return volatilityEstimators

def historicalVolatilityEstimation(df_OHLC, nPeriods, periodsPerYear):
    """
    Returns rolling estimated volatility for nPeriods.
    """
    
    CloseVol = closeToCloseVolatility(df_OHLC, nPeriods, periodsPerYear)
    ParkinsonVol = parkinsonVolatility(df_OHLC, nPeriods, periodsPerYear) 
    GarmanKlassVol = garhmanKlassVolatility(df_OHLC, nPeriods, periodsPerYear) 
    RSVol = rogersSatchellVolatility(df_OHLC, nPeriods, periodsPerYear)
    YZVol = yangZhangVolatility(df_OHLC, nPeriods, periodsPerYear)
    
    volatilityEstimations = pd.DataFrame({
        'Close to Close': CloseVol, 
        'Parkinson': ParkinsonVol, 
        'Garman-Klass': GarmanKlassVol, 
        'Rogers-Satchell' : RSVol, 
        'Yang-Zhang' : YZVol
    })
    
    return volatilityEstimations

# Volatility modelling
def fitGarch11(OHLC, periodsPerYear):
    
    # Format Data
    dataFormatted = pd.DataFrame()
    dataFormatted["Prices"] = OHLC["Adj Close"]
    dataFormatted["Days"] = range(1, len(dataFormatted) + 1)
    dataFormatted["Ui"] = np.log(dataFormatted['Prices'] / dataFormatted['Prices'].shift(1))
    dataFormatted = dataFormatted.dropna()
    dataFormatted['Vi'] = np.nan
    dataFormatted.loc[dataFormatted.index[1], 'Vi'] = dataFormatted['Ui'].iloc[1] ** 2 # Variance initialisation
    dataFormatted['Dates'] = dataFormatted.index
    dataFormatted.set_index('Days', inplace = True)
    
    model = arch_model(dataFormatted['Ui'], p = 1, q = 1, 
                  mean = 'constant', vol = 'GARCH', dist = 'normal', rescale=True)
    
    # Fit the model to the data
    fittedModel = model.fit(disp='off')
    
    # Store results 
    summary = fittedModel.summary()
    
    # Retrieve parameters from the model
    omega = fittedModel.params[1] / (fittedModel.scale ** 2)
    alpha = fittedModel.params[2]
    beta = fittedModel.params[3]
    
    # Compute other parameters
    longRunVariance = (omega / (1 - alpha - beta))
    gamma = omega / longRunVariance
    longRunVolatility = np.sqrt(longRunVariance)
    longRunAnnualizedVolatility = np.sqrt(periodsPerYear) * longRunVolatility
    
    modelResults = {
        "Omega" : omega,
        "Alpha" : alpha,
        "Beta" : beta,
        "Gamma" : gamma,
        "Long Term Ann. Volatility" : longRunAnnualizedVolatility,
        "LongRun Variance" : longRunVariance
    }
    
    return modelResults

def varianceExpectancy(LongTermVariance, alpha, beta, targetPeriods, actVariance):
    """
    Gives variance expectancy for period n + t. 
    Given variance isn't annualized.
    """
    forecastedVariance = LongTermVariance + ((alpha + beta) ** targetPeriods) * (actVariance - V)
    
    return forecastedVariance

def annualizedVolTermStructure(longTermVariance:float, instantaneousVariance:float, alpha:float, beta:float, numberOfPeriods:int, periodsPerYear:int) -> list:
    """
    Returns a list of annualized volatility term structure from period 0 to numberOfPeriods.
    """
    termStructure = []
    
    for i in range (1, numberOfPeriods + 1):
        
        varianceForecast = varianceExpectancy(longTermVariance, alpha, beta, i, instantaneousVariance)
        
        termStructure.append(varianceForecast)
        
    # Converts list to pd.Series, converts variance to volatility and annualizes
    termStructure = np.sqrt(pd.Series(termStructure)) * math.sqrt(periodsPerYear)
    
    return termStructure

def estimateFutureVolatilityRate(longTermVariance, instantaneousVariance, alpha, beta, Period, periodsPerYear):
    """
    Gives mean volatility rate for period from 0 to T. 
    Given volatility is annualized.
    """
    a = math.log(1 / (alpha + beta))

    estimatedVariance = periodsPerYear * (V + ((1 - math.exp(-a*Period)) / (a*Period)) * (V0 - V))

    return math.sqrt(estimatedVariance)

def estimatedFutureVolatilityRateTermStructure(longTermVariance, instantaneousVariance, alpha, beta, maturities):
    """
    Takes a list of differents maturities and estimated annualized volatility term structure.
    Returns a dataframe with each maturity and estimated mean volatility rate.
    
    @param maturities : list of integers.
    """
    
    volatilities = []
    
    for maturity in maturities:
        
        volatility = estimateFutureVolatilityRate(longTermVariance, instantaneousVariance, alpha, beta, maturity)
        volatilities.append(round(volatility, 2))
        
    forecastedTermStructure = pd.DataFrame(volatilities, index = pd.Series(maturities), columns = ["Forecasted Option Volatility %"]).transpose()
    
    return forecastedTermStructure

def adjustmentFactorOverlap(h, T, kind):
    """
    When using overlapping data we need to adjust the data.
    
    @param : h - integer, length of the subseries
    @param : T - integer, total number of observations
    @param : kind - string, vol or var
    """
    n = (T - h) + 1
    
    m = 1 / ((1 - h/n) + ((h**2-1)/(3*n**2)))
    
    if kind == "vol": 
        m = math.sqrt(m)
        
    return m

def showEstimators(firstEstimator, secondEstimator, thirdEstimator, fourthEstimator):
    """
    Takes 4 pd.Series and plots on a figure with 4 subplots.
    """

    # Set the index for the data
    index = firstEstimator.index

    # Create two subplots and unpack the output array immediately
    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(12,8))
    
    linew = 0.75
    labelrot = 75
    linecolor = "turquoise"

    # Plot graphs
    axs[0,0].plot(index, firstEstimator, linewidth=linew, color=linecolor)
    axs[0,0].set_title(f"{firstEstimator.name}")
    axs[0,0].tick_params(axis='x', labelrotation=labelrot)

    axs[0,1].plot(index, secondEstimator, linewidth=linew, color=linecolor)
    axs[0,1].set_title(f"{secondEstimator.name}")
    axs[0,1].tick_params(axis='x', labelrotation=labelrot, color=linecolor)

    axs[1,0].plot(index, thirdEstimator, linewidth=linew, color=linecolor)
    axs[1,0].set_title(f"{thirdEstimator.name}")
    axs[1,0].tick_params(axis='x', labelrotation=labelrot)

    axs[1,1].plot(index, fourthEstimator, linewidth=linew, color=linecolor)
    axs[1,1].set_title(f"{fourthEstimator.name}")
    axs[1,1].tick_params(axis='x', labelrotation=labelrot)

    plt.subplots_adjust(hspace=0.5)
    plt.close(fig)

    return fig

def volatilityCones(OHLC, periods, volatilityEstimator, periodsPerYear):
    """
    Returns volatility cones for a volatility estimator.
    """
    
    adjType = "vol"
    
    # Estimated volatilityies for each rolling period adjusted by the overlapping factor
    estimatedVolatilities = {}
    
    for period in periods:
        
        volatility_estimate = volatilityEstimator(OHLC, period, periodsPerYear)
        
        estimatedVolatilities[period] = volatility_estimate
    
        estimatedVolatilities[period] = estimatedVolatilities[period] * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
    
    estimatedVolatilities = pd.DataFrame(estimatedVolatilities)
    
    # Volatility cones adjusted by the overlapping factor 
    volatilityCones = {}
    
    for period in periods:
        
        Maximum = np.nanquantile(estimatedVolatilities[period], 0.99) * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
        NinetyPercent = np.nanquantile(estimatedVolatilities[period], 0.90) * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
        ThirdQuantile = np.nanquantile(estimatedVolatilities[period], 0.75) * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
        Median = np.nanquantile(estimatedVolatilities[period], 0.5) * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
        Mean = estimatedVolatilities[period].dropna().mean() * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
        FirstQuantile = np.nanquantile(estimatedVolatilities[period], 0.25) * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
        TenPercent = np.nanquantile(estimatedVolatilities[period], 0.10) * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)
        Min = estimatedVolatilities[period].min() * adjustmentFactorOverlap(period, len(estimatedVolatilities[period].dropna()), adjType)

        volatilityCones[period] = [Min, TenPercent, FirstQuantile, Median,Mean, ThirdQuantile, NinetyPercent, Maximum]
    
    volatilityCones = pd.DataFrame(volatilityCones, index = ["Minimum", "10%", '1st Quantile', 'Median', 'Mean', 'Third Quantile',"90%", '99%']).iloc[::-1]
    
    volatilityCones = volatilityCones.transpose()
    
    return {"Estimates" : estimatedVolatilities, 
            "Cones" : volatilityCones}

def showVolatilityCones(volCones):
    """
    Takes the dict returned by volCones and shows volatility cones and associated box plots.
    """
    index = volCones["Cones"].index
    
    # Create two subplots and unpack the output array immediately
    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(12,8), gridspec_kw={'width_ratios': [2, 1]})
    
    linew = 0.75
    # labelrot = 75
    
    # Cones
    axs[0].plot(volCones["Cones"].index, volCones["Cones"], linewidth=linew)
    axs[0].set_title("Volatility Cones")
    axs[0].set_xlabel('Rolling Periods')
    axs[0].set_ylabel('Annualized Volatility')
    
    # Box plots
    axs[1].boxplot(volCones["Estimates"].dropna().values, 
                   labels = volCones["Estimates"].columns, notch=True, 
                   patch_artist=True, boxprops=dict(facecolor = "none", color='cyan'),
                   whiskerprops=dict(color='cyan'), 
                   capprops=dict(color='cyan'),
                   flierprops=dict(markerfacecolor='cyan', marker='.'))
    axs[1].set_title("Volatility Box Plots")
    axs[1].set_xlabel('Rolling Periods')
    axs[1].set_ylabel('Annualized Volatility')
    
    plt.close(fig)

    return fig

def fitGarch11(OHLC, periodsPerYear):
    
    # Format Data
    dataFormatted = pd.DataFrame()
    dataFormatted["Prices"] = OHLC["Adj Close"]
    dataFormatted["Days"] = range(1, len(dataFormatted) + 1)
    dataFormatted["Ui"] = np.log(dataFormatted['Prices'] / dataFormatted['Prices'].shift(1))
    dataFormatted = dataFormatted.dropna()
    dataFormatted['Vi'] = np.nan
    dataFormatted.loc[dataFormatted.index[1], 'Vi'] = dataFormatted['Ui'].iloc[1] ** 2 # Variance initialisation
    dataFormatted['Dates'] = dataFormatted.index
    dataFormatted.set_index('Days', inplace = True)
    
    model = arch_model(dataFormatted['Ui'], p = 1, q = 1, 
                  mean = 'constant', vol = 'GARCH', dist = 'normal', rescale=True)
    
    # Fit the model to the data
    fittedModel = model.fit(disp='off')
    
    # Store results 
    summary = fittedModel.summary()
    
    # Retrieve parameters from the model
    omega = fittedModel.params[1] / (fittedModel.scale ** 2)
    alpha = fittedModel.params[2]
    beta = fittedModel.params[3]
    
    # Compute other parameters
    longRunVariance = (omega / (1 - alpha - beta))
    gamma = omega / longRunVariance
    longRunVolatility = np.sqrt(longRunVariance)
    longRunAnnualizedVolatility = np.sqrt(periodsPerYear) * longRunVolatility
    
    modelResults = {
        "Omega" : omega,
        "Alpha" : alpha,
        "Beta" : beta,
        "Gamma" : gamma,
        "Long Term Ann. Volatility" : longRunAnnualizedVolatility,
        "LongRun Variance" : longRunVariance
    }
    
    return modelResults

def varianceExpectancy(LongTermVariance, alpha, beta, targetPeriods, actVariance):
    """
    Gives variance expectancy for period n + t. 
    Given variance isn't annualized.
    """
    forecastedVariance = LongTermVariance + ((alpha + beta) ** targetPeriods) * (actVariance - V)
    
    return forecastedVariance

def annualizedVolTermStructure(longTermVariance:float, instantaneousVariance:float, alpha:float, beta:float, numberOfPeriods:int, periodsPerYear:int) -> list:
    """
    Returns a list of annualized volatility term structure from period 0 to numberOfPeriods.
    """
    termStructure = []
    
    for i in range (1, numberOfPeriods + 1):
        
        varianceForecast = varianceExpectancy(longTermVariance, alpha, beta, i, instantaneousVariance)
        
        termStructure.append(varianceForecast)
        
    # Converts list to pd.Series, converts variance to volatility and annualizes
    termStructure = np.sqrt(pd.Series(termStructure)) * math.sqrt(periodsPerYear)
    
    return termStructure

def estimateFutureVolatilityRate(longTermVariance, instantaneousVariance, alpha, beta, Period, periodsPerYear):
    """
    Gives mean volatility rate for period from 0 to T. 
    Given volatility is annualized.
    """
    a = math.log(1 / (alpha + beta))

    estimatedVariance = periodsPerYear * (V + ((1 - math.exp(-a*Period)) / (a*Period)) * (V0 - V))

    return math.sqrt(estimatedVariance)

def estimatedFutureVolatilityRateTermStructure(longTermVariance, instantaneousVariance, alpha, beta, maturities):
    """
    Takes a list of differents maturities and estimated annualized volatility term structure.
    Returns a dataframe with each maturity and estimated mean volatility rate.
    
    @param maturities : list of integers.
    """
    
    volatilities = []
    
    for maturity in maturities:
        
        volatility = estimateFutureVolatilityRate(longTermVariance, instantaneousVariance, alpha, beta, maturity)
        volatilities.append(round(volatility, 2))
        
    forecastedTermStructure = pd.DataFrame(volatilities, index = pd.Series(maturities), columns = ["Forecasted Option Volatility %"]).transpose()
    
    return forecastedTermStructure

def adjustmentFactorOverlap(h, T, kind):
    """
    When using overlapping data we need to adjust the data.
    
    @param : h - integer, length of the subseries
    @param : T - integer, total number of observations
    @param : kind - string, vol or var
    """
    n = (T - h) + 1
    
    m = 1 / ((1 - h/n) + ((h**2-1)/(3*n**2)))
    
    if kind == "vol": 
        m = math.sqrt(m)
        
    return m

# Intermarket volatility


# Risk Analysis



# Positions risks


# Option pricing models

class BlackScholesOption:
    """
    Black & Scholes option object for european style options.
    """
    
    def __init__(self, S:float, K:float, T:float, r:float, q:float, sigma: float, option_type: str, option_style:str):
    
        assert sigma >=0, "Volatility can't be less than zero"
        assert S >=0, "Initial stock value can't be less than zero"
        assert K >=0, "Strike price can't be less than zero"
        assert T >=0, "Time to maturity can't be less than zero"
        assert option_type in ['Call', 'Put'], "Option type must be either Call or Put"
        assert option_style == 'European', "Option style must be European"
        assert q >=0, "Dividend yield cannot be less than zero"
        
        # Parameters
        self.type = str(option_type)
        self.style = str(option_style)
        self.S = float(S)
        self.K = float(K)
        self.r = float(r) / 100
        self.sigma = float(sigma) / 100
        self.T = float(T) / 365
        self.q = float(q) / 100
        
        # Calculations
        self.d1 = self.d1()
        self.d2 = self.d2()
        
        self.price = self.optionPrice()
        
        self.delta = self.optionDelta()
        self.gamma = self.optionGamma()
        self.vega = self.optionVega()
        self.theta = self.optionTheta()
        self.rho = self.optionRho()
        
        self.vanna = self.optionVanna()
        self.volga = self.optionVolga()
        self.charm = self.optionCharm()
        self.color = self.optionColor()
        self.speed = self.optionSpeed()
        
        self.gearing  = self.optionLambda()
        self.epsilon = self.optionEpsilon()
        
    def N(self, x):
        """
        Cumulative distribution function of a normal distribution.
        """
        return norm.cdf(x, 0, 1)

    def n(self, x):
        """
        Probability density function of a normal distribution.
        """
        return norm.pdf(x, 0, 1)

    def d1(self):
        d1 = (math.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T)/ \
        (self.sigma * math.sqrt(self.T))

        return d1

    def d2(self):
        d2 = self.d1 - self.sigma * math.sqrt(self.T)

        return d2

    def optionPrice(self) -> float:
        if self.type == 'Put':
            price = self.K * math.exp(-self.r * self.T) * self.N(-self.d2) - self.S * math.exp(-self.q * self.T) * self.N(-self.d1)
        elif self.type == 'Call':
            price = self.S * math.exp(-self.q * self.T) * self.N(self.d1) - self.K * math.exp(-self.r * self.T) * self.N(self.d2)

        return price

    def optionDelta(self) -> float:
        """
        Rate of change of the theoretical option value with respect to changes in the underlying price.
        """
        if self.type == 'Put':
            delta = -math.exp(-self.q * self.T) * self.N(-self.d1)
        elif self.type == 'Call':
            delta = math.exp(-self.q * self.T) * self.N(self.d1)

        return delta

    def optionGamma(self) -> float:
        """
        Rate of change in the delta with respect to changes in the underlying price.
        """
        gamma = (self.n(self.d1) * math.exp(-self.q * self.T)) / (self.S * self.sigma * math.sqrt(self.T))

        return gamma

    def optionVega(self) -> float:
        """
        Sensivity of the option theoretical value with respect to changes in the volatility.
        """
        vega = self.S * math.exp(-self.q * self.T) * self.n(self.d1) * math.sqrt(self.T)

        return vega * 0.01

    def optionTheta(self) -> float:
        """
        Sensivity of the option theoretical value over the passage of time. 
        """
        if self.type == 'Put':
            theta = -math.exp(-self.q * self.T) * (self.S * self.n(self.d1) * self.sigma) / (2 * math.sqrt(self.T)) + self.r * \
            self.K * math.exp(-self.r * self.T) * self.N(-self.d2) - self.q * self.S * math.exp(-self.q * self.T) * self.N(-self.d1)
        elif self.type == 'Call':
            theta = - math.exp(-self.q * self.T) * (self.S * self.n(self.d1) * self.sigma) / (2 * math.sqrt(self.T)) - self.r * \
            self.K * math.exp(-self.r * self.T) * self.N(self.d2) + self.q * self.S * math.exp(-self.q * self.T) * self.N(self.d1)

        return theta / 365

    def optionRho(self) -> float:
        """
        Sensivity of the option price to changes in interest rates. 
        """
        if self.type == 'Put':
            rho = -self.K * self.T * math.exp(-self.r * self.T) * self.N(-self.d2)
        elif self.type == 'Call':
            rho = self.K * self.T * math.exp(-self.r * self.T)*self.N(self.d2)
        return rho * 0.01

    def optionLambda(self) -> float:
        """
        Lambda, omega or elasticity. Called gearing.
        Represents the percentage change in option value per percentage change in the underlying price.
        """

        gearing = self.delta * (self.S / self.price)

        return gearing

    def optionEpsilon(self) -> float:
        """
        Epsilon. Called psi.
        Represents the percentage change in option value per percentage change in the underlying dividend yield. 
        """
        if self.type == 'Put':
            epsilon = self.S * self.r * math.exp(-self.q * self.T) * self.N(-self.d1)
        elif self.type == 'Call':
            epsilon = - self.S * self.r * math.exp(-self.q * self.T) * self.N(self.d1)

        return epsilon

    def optionVanna(self) -> float:
        """
        Sensitivity of the option delta with respect to change in volatility or,
        alternatively, the partial of vega with respect to the underlying instrument's price.
        """
        vanna = - math.exp(-self.q * self.T) * self.n(self.d1) * (self.d2 / self.sigma)

        return vanna * 0.01

    def optionVolga(self) -> float:
        """
        Vomma or Volga measures the rate of change to vega as volatility changes. 
        It is the second order price sensivity to volatility. 
        """
        volga = self.S * math.exp(-self.q * self.T) * self.n(self.d1) * math.sqrt(self.T) * (self.d1 * self.d2 / self.sigma)

        return volga * 0.0001

    def optionCharm(self) -> float:
        """
        Charm is a second-order derivative of the option value, once to price and once to the passage of time. 
        It is also then the derivative of theta with respect to the underlying's price.
        """
        if self.type == 'Put':
            charm = -self.q * math.exp(-self.q * self.T) * self.N(-self.d1) - math.exp(-self.q * self.T) * self.n(self.d1) * \
            (2 * (self.r - self.q) * self.T - self.d2 * self.sigma * math.sqrt(self.T)) / (2 * self.T * self.sigma * math.sqrt(self.T))
        elif self.type == 'Call':
            charm = self.q * math.exp(-self.q * self.T) * self.N(self.d1) - math.exp(-self.q * self.T) * self.n(self.d1) * \
            (2 * (self.r - self.q) * self.T - self.d2 * self.sigma * math.sqrt(self.T)) / (2 * self.T * self.sigma * math.sqrt(self.T))

        return charm / 365

    def optionColor(self) -> float:
        """
        Rate of change in the gamma over the passage of time.
        """
        color = -math.exp(-self.q * self.T) * (self.n(self.d1) / (2 * self.S * self.T * self.sigma * math.sqrt(self.T))) * \
        (2 * self.q * self.T + 1 + ((2 * (self.r - self.q) * self.T - self.d2 * self.sigma * math.sqrt(self.T)) / (self.sigma * math.sqrt(self.T))) * self.d1)

        return color / 365

    def optionSpeed(self) -> float:
        """
        Rate of change in the gamma with respect to changes in the underlying price.
        """
        speed = -math.exp(-self.q * self.T) * (self.n(self.d1) / (self.S ** 2 * self.sigma * math.sqrt(self.T))) * \
        ((self.d1 / (self.sigma * math.sqrt(self.T))) + 1)

        return speed

    def impliedVolatility(self, marketPrice, tolerance = 0.00001):
        """
        Derives the implied volatility of an European Option with Newton-Raphson Algorithm
        """
        S = self.S
        K = self.K
        T = self.T * 365
        r = self.r * 100
        q = self.q * 100
        oldVolatility = self.sigma * 100

        maxIterations = 500

        for k in range(maxIterations):
            
            newInstance = BlackScholesOption(S, K, T, r, q, oldVolatility, self.type, self.style)
            theoPrice = newInstance.price
            optionPremium = newInstance.vega * 100
            OptionPrice = theoPrice - marketPrice
            newVolatility = oldVolatility - OptionPrice / optionPremium
            
            if newVolatility < 0:
                continue
            
            newTheoPrice = BlackScholesOption(S, K, T, r, q, newVolatility, self.type, self.style).price

            if (abs(oldVolatility - newVolatility) < tolerance or abs(newTheoPrice - marketPrice) < tolerance):
                break
            oldVolatility = newVolatility

        impliedVolatility = oldVolatility

        return impliedVolatility
    
    def optionPayoff(self, St):
    
        if self.type == 'Call':

            payoff = max(St - self.K, 0)

        elif self.type == 'Put':

            payoff = max(self.K - St, 0)

        return payoff
    
class BinomialModelOption:
    
    def __init__(self, S0:float, K:float, T:float, r:float, q:float, sigma: float, N:int, option_type: str, option_style: str):
        
        assert sigma >=0, "Volatility can't be less than zero"
        assert S0 >=0, "Initial stock value can't be less than zero"
        assert T >=0, "Time to maturity can't be less than zero"
        assert N <=1000, "Maximum authorized steps : 1000"
        assert N >=1, "Need at least 1 time step"
        assert option_type in ['Call', 'Put'], "Option type must be either Call or Put"
        assert option_style in ['European', 'American'], "Option style must be either European or American"
        assert q >=0, "Dividend yield cannot be less than zero"

        # Parameters
        self.type = str(option_type)
        self.style = str(option_style)
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T) / 365
        self.r = float(r) / 100
        self.q = float(q) / 100
        self.sigma = float(sigma) / 100
        self.N = int(N)

        # Calculations
        self.dt = self.T / self.N
        self.u = self.uCalc()
        self.d = self.dCalc()
        self.p = self.pCalc()

        # Pricing
        self.price = self.optionPrice()

        # Greeks
        self.delta = self.optionDelta()
        self.gamma = self.optionGamma()
        self.vega = self.optionVega()
        self.theta = self.optionTheta()
        self.rho = self.optionRho()
        self.charm = self.optionCharm()
        self.vanna = self.optionVanna()
        self.volga = self.optionVolga()
        self.color = self.optionColor()
        
    def uCalc(self, sigma = None):
        
        if sigma is None:
            sigma = self.sigma
            
        return math.exp(sigma * math.sqrt(self.dt))

    def dCalc(self, sigma = None):
        
        if sigma is None:
            sigma = self.sigma
        
        return math.exp(-sigma * math.sqrt(self.dt))

    def pCalc(self, u = None, d = None):
        
        if u is None and d is None:
            u = self.u
            d = self.d

        return (math.exp((self.r - self.q) * self.dt) - d )  /  (u - d)
    
    def udpUpdate(self, sigma):
        self.u = self.uCalc(sigma)
        self.d = self.dCalc(sigma)
        self.p = self.pCalc(u = self.u, d = self.d)

    def optionPrice(self, S0=None):
        
        if S0 is None:
            S0 = self.S0
        
        # Stock Prices - Can use numpy to vectorize. 
        stock_prices_tree = np.zeros((self.N + 1, self.N + 1)) # Initialise tree
        stock_prices_tree[0][0] = S0 # Set first value

        for i in range(1, self.N + 1):
            stock_prices_tree[i, 0] = stock_prices_tree[i - 1,0] * self.u # Only ups
            
            for j in range(1, i + 1):
                stock_prices_tree[i, j] = stock_prices_tree[i - 1, j - 1] * self.d
        
        # Option Prices
        option_prices_tree = np.zeros((self.N + 1, self.N + 1))

        # Backward recursion - optimize payoff function to faster calculations. Can also use numpy to vectorize. 
        for j in range(self.N + 1): # Last node
            if self.type == 'Call':
                option_prices_tree[self.N, j] = max(0, stock_prices_tree[self.N, j] - self.K)
            elif self.type == 'Put':
                option_prices_tree[self.N, j] = max(0, self.K - stock_prices_tree[self.N, j])

        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                    option_prices_tree[i, j] = math.exp(-self.r * self.dt) * (self.p * option_prices_tree[i + 1, j] + (1 - self.p) * option_prices_tree[i + 1, j + 1])
                    if self.style == "American":
                        if self.type == 'Call':
                            option_prices_tree[i, j] = max(option_prices_tree[i, j], stock_prices_tree[i, j] - self.K)
                        elif self.type == 'Put':
                            option_prices_tree[i, j] = max(option_prices_tree[i, j], self.K - stock_prices_tree[i, j])
        
        self.tree = option_prices_tree
        self.stock_tree = stock_prices_tree
        
        return option_prices_tree[0][0]

    def optionDelta(self):
        """
        Computes delta using the binomial method.
        """
        
        delta = (self.tree[1][0] - self.tree[1][1]) / (self.S0 * (self.u - self.d))
        
        return delta
      
    def optionGamma(self):
        """
        Computes the gamma using the binomial method.
        """
        
        fstnum = (self.tree[2][0] - self.tree[2][1]) / (self.stock_tree[2][0] - self.stock_tree[2][1])
        scndnum = (self.tree[2][1] - self.tree[2][2]) / (self.stock_tree[2][1] - self.stock_tree[2][2])
        den = 0.5 * (self.stock_tree[2][0] - self.stock_tree[2][2])

        gamma = (fstnum - scndnum) / den
        
        return gamma
    

    def optionVega(self):
        
        dv = 0.01 # 1% volatility change
        
        initial_vol = self.sigma
        
        backward_vol = initial_vol - dv
        forward_vol = initial_vol + dv
        
        # Backward option price
        self.sigma = backward_vol
        self.udpUpdate(sigma = self.sigma)
        backward_price = self.optionPrice()
        
        # Forward option price
        self.sigma = forward_vol
        self.udpUpdate(sigma = self.sigma)
        forward_price = self.optionPrice()
        
        self.sigma = initial_vol
        self.udpUpdate(sigma = self.sigma)
        
        vega = (forward_price - backward_price) / (2 * dv)
        
        return vega / 100

    def optionTheta(self):
        
        theta = -(self.tree[0][0] - self.tree[2][1]) / (2 * (self.dt * 365))
        
        return theta

    def optionRho(self):
        
        dv = 0.01
        
        initial_rate = self.r
        
        backward_rate = initial_rate - dv
        forward_rate = initial_rate + dv
        
        # Backward option price
        self.r = backward_rate
        self.p = self.pCalc()
        backward_price = self.optionPrice()
        
        # Forward option price
        self.r = forward_rate
        self.p = self.pCalc()
        forward_price = self.optionPrice()
        
        self.r = initial_rate
        self.p = self.pCalc()
        
        rho = (forward_price - backward_price) / (2 * dv)
        
        return rho / 100
    
    def optionCharm(self):
        pass
    
    def optionColor(self):
        pass

    def optionVanna(self):
        pass

    def optionVolga(self):
        pass

    def payoffOption(self, param1, rng1, param2, rng2):
        pass