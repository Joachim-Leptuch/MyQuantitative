from typing import Optional, Literal

# Option Object
class Option:

    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        sigma: float,
        q: float,
        option_type: Literal['call', 'put'],
        exercise_style: Literal['European', 'American', 'Bermudan'],
        exotic_feature: Optional[str] = None
    ):

        # Input validation
        assert s >= 0, "Initial stock value can't be less than zero"
        assert k >= 0, "Strike price can't be less than zero"
        assert t >= 0, "Time to maturity can't be less than zero"
        assert sigma >= 0, "Volatility can't be less than zero"
        assert q >= 0, "Dividend yield cannot be negative"
        assert option_type in ['call', 'put'], "Option type must be either Call or Put"
        assert exercise_style in ['European', 'American', 'Bermudan'], "Invalid exercise style"

        # Store input parameters
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type
        self.exercise_style = exercise_style
        self.exotic_feature = exotic_feature

        # Initialize Greeks and model parameters
        self._model: Optional[str] = None
        self._price: Optional[float] = None
        self._intrinsic_value: Optional[float] = None
        self._time_value: Optional[float] = None

        self._delta: Optional[float] = None
        self._gamma: Optional[float] = None
        self._vega: Optional[float] = None
        self._theta: Optional[float] = None
        self._rho: Optional[float] = None

        self._vanna: Optional[float] = None
        self._volga: Optional[float] = None
        self._charm: Optional[float] = None
        self._color: Optional[float] = None
        self._speed: Optional[float] = None

        self._gearing: Optional[float] = None
        self._epsilon: Optional[float] = None