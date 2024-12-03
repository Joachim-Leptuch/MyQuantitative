from functools import cached_property
from typing import Optional, Literal
import math
from scipy.stats import norm


class BlackScholesOption(Option):

        def __init__(
                self,
                s: float,           # Current spot price
                k: float,           # Option strike price
                t: float,           # Option maturity in years
                r: float,           # Risk-free rate
                sigma: float,       # Underlying volatility
                q: float = 0.0,     # Underlying continuous dividend rate
                option_type: Literal['call', 'put'] = 'call'
        ):
            # Call parent class constructor with European exercise style
            super().__init__(
                s=s,
                k=k,
                t=t,
                r=r,
                sigma=sigma,
                q=q,
                option_type=option_type,
                exercise_style='European'
            )

            # Set the model type
            self._model = 'Black-Scholes'

            # Calculate price and Greeks
            self._calculate_price()
            self._calculate_greeks()

        @cached_property
        def d1(self) -> float:
            """
            Calculate d1 parameter for Black-Scholes model using cached property.
            """
            return (math.log(self.s / self.k) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.t) / (
                        self.sigma * math.sqrt(self.t))

        @cached_property
        def d2(self) -> float:
            """
            Calculate d2 parameter for Black-Scholes model using cached property.
            """
            return self.d1 - self.sigma * math.sqrt(self.t)

        def _calculate_price(self):
            """
            Calculate option price using Black-Scholes formula.
            Uses cached d1 and d2 values.
            """
            if self.option_type == 'call':
                self._price = (
                        self.s * math.exp(-self.q * self.t) * norm.cdf(self.d1) -
                        self.k * math.exp(-self.r * self.t) * norm.cdf(self.d2)
                )
                self._intrinsic_value = max(0, self.s - self.k)
            else:  # put
                self._price = (
                        self.k * math.exp(-self.r * self.t) * norm.cdf(-self.d2) -
                        self.s * math.exp(-self.q * self.t) * norm.cdf(-self.d1)
                )
                self._intrinsic_value = max(0, self.k - self.s)

            self._time_value = self._price - self._intrinsic_value

        def _calculate_greeks(self):
            """
            Calculate option Greeks for Black-Scholes model.
            Uses cached d1 and d2 values.
            """
            # Delta
            if self.option_type == 'call':
                self._delta = math.exp(-self.q * self.t) * norm.cdf(self.d1)
            else:
                self._delta = -math.exp(-self.q * self.t) * norm.cdf(-self.d1)

            # Gamma
            self._gamma = (
                    math.exp(-self.q * self.t) * norm.pdf(self.d1) /
                    (self.s * self.sigma * math.sqrt(self.t))
            )

            # Vega
            self._vega = (
                    self.s * math.exp(-self.q * self.t) *
                    norm.pdf(self.d1) * math.sqrt(self.t)
            )

            # Theta (approximation)
            if self.option_type == 'call':
                self._theta = -(
                        self.s * math.exp(-self.q * self.t) * norm.pdf(self.d1) * self.sigma /
                        (2 * math.sqrt(self.t)) -
                        self.r * self.k * math.exp(-self.r * self.t) * norm.cdf(self.d2) +
                        self.q * self.s * math.exp(-self.q * self.t) * norm.cdf(self.d1)
                )
            else:
                self._theta = -(
                        self.s * math.exp(-self.q * self.t) * norm.pdf(self.d1) * self.sigma /
                        (2 * math.sqrt(self.t)) +
                        self.r * self.k * math.exp(-self.r * self.t) * norm.cdf(-self.d2) -
                        self.q * self.s * math.exp(-self.q * self.t) * norm.cdf(-self.d1)
                )

            # Rho
            if self.option_type == 'call':
                self._rho = self.k * self.t * math.exp(-self.r * self.t) * norm.cdf(self.d2)
            else:
                self._rho = -self.k * self.t * math.exp(-self.r * self.t) * norm.cdf(-self.d2)

            # Vanna
            self._vanna = - math.exp(-self.q * self.t) * self.pdf(self._d1) * (self._d2 / self.sigma) * 0.01

            # Volga
            self._volga = (self.s * math.exp(-self.q * self.t) * self.pdf(self._d1) * math.sqrt(self.t) *
                          (self._d1 * self._d2 / self.sigma)) * 0.0001

        # Implied volatility

        @property
        def price(self) -> float:
            """
            Getter for option price.
            """
            return self._price

        @property
        def greeks(self) -> dict:
            """
            Getter for all calculated Greeks.
            """
            return {
                'delta': self._delta,
                'gamma': self._gamma,
                'vega': self._vega,
                'theta': self._theta,
                'rho': self._rho,
                'vanna': self._vanna,
                'volga': self._volga
            }