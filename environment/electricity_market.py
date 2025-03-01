import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils

from .reward_types import RewardType, DemandType

class ElectricityMarketEnv(gym.Env):
    """
    An environment for simulating an electricity market with a battery storage system.

    The agent interacts with the environment by taking continuous actions,
    charging (positive) or discharging (negative) a battery.

    States:
        - SoC: State of Charge - current energy level in the battery.
        - D_t: Household electricity demand at the current timestep.
        - P_t: Market price of electricity at the current timestep.

    Action:
        A continuous value in [-battery_cap, battery_cap], the amount of
        charging/discharging the battery.
    """

    def __init__(self, **kwargs):
        super(ElectricityMarketEnv, self).__init__()

        params = kwargs.get("params", {})

        # Battery parameters
        self.battery_capacity = params.get("battery_capacity", 100)
        assert self.battery_capacity > 0, "Battery capacity must be positive."
        self.initial_soc = params.get("initial_soc", 50)
        assert self.initial_soc <= self.battery_capacity, f"Initial SoC {self.initial_soc} exceeds battery capacity {self.battery_capacity}."
        self.soc = self.initial_soc  # Current SoC

        # Environment dynamics parameters
        self.max_steps = params.get("max_steps", 200)  # Total timesteps in one episode
        self.current_step = 0

        # action space
        self.action_space = spaces.Box(
            low=-self.battery_capacity,
            high=self.battery_capacity,
            shape=(1,),
            dtype=np.float32
        )

        # state space: [SoC, Demand, Price] all dimensions are continuous
        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([
            self.battery_capacity,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.reward_type = params.get("reward type", RewardType.PROFIT)
        assert isinstance(self.reward_type, RewardType), f"Invalid reward type."

        self.demand_type = params.get("demand type", DemandType.GAUSSIAN)
        assert isinstance(self.demand_type, DemandType), f"Invalid demand type."

        self.seed()

    def seed(self, seed=None):
        """Set the seed for random number generation."""
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state at the start of an episode."""
        if seed is not None:
            self.seed(seed)

        self.soc = self.initial_soc
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """Returns current observation [SoC, demand, price]."""
        t_norm = self.current_step / self.max_steps  # Normalized time [0, 1]
        demand = self._demand_function(t_norm)
        price = self._price_function(demand, t_norm)

        # Ensure scalar values
        if isinstance(demand, np.ndarray):
            demand = demand.item() if demand.size == 1 else float(demand[0])

        if isinstance(price, np.ndarray):
            price = price.item() if price.size == 1 else float(price[0])

        obs = np.array([float(self.soc), float(demand), float(price)], dtype=np.float32)
        return obs

    def _demand_function(self, t):
        """Compute the household electricity demand at time t."""
        demand = 0.0

        if self.demand_type == DemandType.GAUSSIAN:
            demand = 80 * np.exp(-((t - 0.3) ** 2) / (2 * 0.05**2)) + \
                     120 * np.exp(-((t - 0.75) ** 2) / (2 * 0.08**2))

        elif self.demand_type == DemandType.SINUSOIDAL:
            demand = 80 * np.sin(4 * np.pi * t - 2) + 30 * np.sin(2 * np.pi * t) + 70

        else:  # DemandType.STEP
            if 0.2 <= t < 0.35:
                demand = 100
            elif 0.7 <= t < 0.85:
                demand = 150
            else:
                demand = 0

        noise = float(self.np_random.normal(0, 5.0))
        demand = max(float(demand) + noise, 0.0)  # Ensure demand is non-negative and a scalar

        return demand

    def _price_function(self, demand, t):
        """
        Compute the market price of electricity at time t.
        The price is a linear transformation of the demand.
        """
        if isinstance(demand, np.ndarray):
            demand = demand.item() if demand.size == 1 else float(demand[0])

        m, b = 0.5, 10.0  # Base price plus demand component
        base_price = m * float(demand) + b
        noise = float(self.np_random.normal(0, 2.0))
        price = max(base_price + noise, 0.0)

        return float(price)  # Ensure returned price is a scalar

    def step(self, action):
        """
        Execute one timestep in the environment.

        Parameters:
            action (array): Representing the amount of energy to charge or discharge.

        Returns:
            obs (array): Next observation ([SoC, demand, price]).
            reward (float): Reward earned this timestep.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            info (dict): Additional info (empty in this implementation).
        """
        # Clip action within allowed bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_value = float(action[0].item())  # Ensure scalar
        info = {}

        # Compute current demand and price based on normalized time
        t_norm = self.current_step / self.max_steps
        demand = self._demand_function(t_norm)
        price = self._price_function(demand, t_norm)

        # Ensure demand and price are scalar values
        if isinstance(demand, np.ndarray):
            demand = demand.item() if demand.size == 1 else float(demand[0])

        if isinstance(price, np.ndarray):
            price = price.item() if price.size == 1 else float(price[0])

        if action_value >= 0:
            # Charging the battery
            charge_amount = min(action_value, self.battery_capacity - self.soc)
            cost = price * charge_amount
            self.soc += charge_amount
            reward = -float(cost)

        else:
            # Discharging the battery
            discharge_requested = -action_value
            discharge_possible = min(discharge_requested, self.soc)
            self.soc -= discharge_possible

            # Use discharged energy to meet the household demand
            # Any energy beyond meeting the demand is sold to the grid
            if discharge_possible >= demand:
                sold_energy = discharge_possible - demand
            else:  # Not enough to cover demand
                sold_energy = 0.0

            # Compute reward
            reward = self._compute_reward(demand, sold_energy, price, discharge_possible)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Construct next observation - ensure all values are scalar
        obs = np.array([float(self.soc), float(demand), float(price)], dtype=np.float32)

        return obs, float(reward), False, done, info

    def _compute_reward(self, demand, sold_energy, price, discharge_amount):
        """
        Compute the reward based on the reward type.

        Parameters:
            demand (float): The household energy demand.
            sold_energy (float): The amount of energy sold.
            price (float): The price at which energy is sold.
            discharge_amount (float): The amount of energy discharged.

        Returns:
            reward (float): The computed reward.
        """
        if self.reward_type == RewardType.PROFIT:
            # Use just the sold energy for profit calculation
            reward = float(price) * float(sold_energy)

        else:  # RewardType.INTERNAL_DEMAND
            penalty = 10.0
            unmet_demand = max(0.0, float(demand) - float(discharge_amount))
            # Balanced reward calculation
            reward = float(price) * float(sold_energy) - penalty * float(unmet_demand)

        return float(reward)

    def render(self, mode='human'):
        """Render the current state of the environment."""
        print(f"Step: {self.current_step}, SoC: {self.soc:.2f}")