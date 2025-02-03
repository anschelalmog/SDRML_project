import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
from rl_project.utils import  get_logger

logger = get_logger()

class ElectricityMarketEnv(gym.Env):
    """
    A custom Gym environment for simulating an electricity market with a battery storage system.

    The agent can take continuous actions representing charging (positive values) or discharging (negative values)
    of a battery. The environment models the dynamics of household electricity demand and market price, both of which
    follow periodic functions with noise. The reward is computed based on the selected reward function.

    State:
        - SoC: State of Charge (current energy level in the battery)
        - Dt: Household electricity demand at the current timestep
        - Pt: Market price of electricity at the current timestep

    Action:
        - Continuous value in [-battery_capacity, battery_capacity]
            Positive action: Charging the battery.
            Negative action: Discharging the battery.

    Reward (Options):
        1. "profit" (default):
           Reward = \( P_t \times \text{sold\_energy} \)
           Rewards the agent solely based on the revenue generated from selling surplus energy after meeting demand.

        2. "penalty_unmet":
           Reward = \( P_t \times (\text{discharge\_amount}) - \lambda \times \max(0, D_t - \text{discharge\_amount}) \)
           Rewards all discharged energy but applies a penalty for any portion of household demand that is not met.
           (This encourages the agent to prioritize internal demand.)

        3. "degradation":
           Reward = \( P_t \times (\text{sold\_energy}) - c \times (\text{discharge\_amount}) \)
           Rewards surplus energy sold while subtracting a cost proportional to the total discharged energy to account for battery degradation.
    """

    def __init__(self, args=None):
        super(ElectricityMarketEnv, self).__init__()

        # --------------------------
        # Battery parameters
        # --------------------------
        self.battery_capacity = 100.0  # Maximum energy that can be stored in the battery (e.g., in kWh)
        self.initial_soc = 50.0        # Initial State of Charge (SoC)
        self.soc = self.initial_soc    # Current SoC

        # --------------------------
        # Environment dynamics parameters
        # --------------------------
        self.max_steps = 200           # Total timesteps in one episode
        self.current_step = 0          # Timestep counter

        # --------------------------
        # Define the action space.
        # The agent's action is a continuous value in [-battery_capacity, battery_capacity].
        # A positive value represents charging and a negative value represents discharging.
        # --------------------------
        self.action_space = spaces.Box(low=-self.battery_capacity,
                                       high=self.battery_capacity,
                                       shape=(1,),
                                       dtype=np.float32)

        # --------------------------
        # Define the observation space.
        # The state consists of [SoC, Demand, Price].
        # SoC is bounded between 0 and battery_capacity.
        # Demand and Price are non-negative; we use a very high upper bound.
        # --------------------------
        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([self.battery_capacity, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --------------------------
        # Reward function selection.
        # Default is "profit", but you can choose "penalty_unmet" or "degradation" via the args parameter.
        # --------------------------
        self.reward_type = "profit"
        if args is not None and hasattr(args, 'reward_type'):
            self.reward_type = args.reward_type
            logger.info(f"ElectricityMarketEnv initialized with battery_capacity={self.battery_capacity}, initial_soc={self.initial_soc}, max_steps={self.max_steps}, reward_type={self.reward_type}")

    # --------------------------
        # Set the random seed for reproducibility.
        # --------------------------
        self.seed()

    def seed(self, seed=None):
        """
        Set the seed for random number generation.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment to the initial state at the start of an episode.
        """
        self.soc = self.initial_soc
        self.current_step = 0
        logger.info(f"Environment reset: initial SoC={self.soc}")
        return self._get_obs()

    def _get_obs(self):
        """
        Construct the current observation.
        The observation includes the current SoC, the demand, and the price.
        Demand and price are functions of normalized time.
        """
        t_norm = self.current_step / self.max_steps  # Normalized time [0, 1]
        demand = self._demand_function(t_norm)
        price = self._price_function(t_norm)
        obs = np.array([self.soc, demand, price], dtype=np.float32)
        logger.debug(f"Observation: SoC={self.soc}, Demand={demand:.2f}, Price={price:.2f}")
        return obs

    def _demand_function(self, t):
        """
        Compute the household electricity demand at time t.

        The demand function is modeled as a combination of two Gaussian functions:

        \[
        f_D(t) = 100 \cdot \exp\left(-\frac{(t-0.4)^2}{2 \cdot (0.05)^2}\right)
                + 120 \cdot \exp\left(-\frac{(t-0.7)^2}{2 \cdot (0.1)^2}\right)
        \]

        A small Gaussian noise (mean 0, std 5.0) is added to simulate randomness.
        """
        base_demand = (100.0 * np.exp(-((t - 0.4)**2) / (2 * (0.05**2))) +
                       120.0 * np.exp(-((t - 0.7)**2) / (2 * (0.1**2))))
        noise = self.np_random.normal(0, 5.0)  # Noise term
        demand = max(base_demand + noise, 0.0)   # Ensure demand is non-negative
        logger.debug(f"Computed demand={demand:.2f} at t={t:.2f}")

        return demand

    def _price_function(self, t):
        """
        Compute the market price of electricity at time t.

        The price function is modeled as a combination of two Gaussian functions:

        \[
        f_P(t) = 50 \cdot \exp\left(-\frac{(t-0.3)^2}{2 \cdot (0.07)^2}\right)
                + 80 \cdot \exp\left(-\frac{(t-0.8)^2}{2 \cdot (0.08)^2}\right)
        \]

        A small Gaussian noise (mean 0, std 2.0) is added to simulate randomness.
        """
        base_price = (50.0 * np.exp(-((t - 0.3)**2) / (2 * (0.07**2))) +
                      80.0 * np.exp(-((t - 0.8)**2) / (2 * (0.08**2))))
        noise = self.np_random.normal(0, 2.0)  # Noise term
        price = max(base_price + noise, 0.0)     # Ensure price is non-negative
        logger.debug(f"Computed price={price:.2f} at t={t:.2f}")
        return price

    def step(self, action):
        """
        Execute one timestep in the environment.

        Parameters:
            action (array): A 1D numpy array with one element representing the
                            amount of energy to charge (positive) or discharge (negative).

        Returns:
            obs (array): Next observation ([SoC, demand, price]).
            reward (float): Reward earned this timestep.
            done (bool): Whether the episode has ended.
            info (dict): Additional info (empty in this implementation).
        """
        # Clip the action within allowed bounds.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_value = action[0]  # Extract scalar from array
        info = {}

        # Compute current demand and price based on normalized time.
        t_norm = self.current_step / self.max_steps
        demand = self._demand_function(t_norm)
        price = self._price_function(t_norm)

        if action_value >= 0:
            # --------------------------
            # Charging the battery.
            # The battery cannot be charged beyond its capacity.
            # --------------------------
            charge_amount = min(action_value, self.battery_capacity - self.soc)
            self.soc += charge_amount
            reward = 0.0  # No immediate reward for charging
            logger.info(f"Step {self.current_step}: Charging battery by {charge_amount:.2f} kWh, New SoC={self.soc:.2f}")

        else:
            # --------------------------
            # Discharging the battery.
            # First, limit the discharge to the available energy in the battery.
            # --------------------------
            discharge_requested = -action_value  # Convert to a positive value
            discharge_possible = min(discharge_requested, self.soc)
            self.soc -= discharge_possible

            # --------------------------
            # Use discharged energy to meet the household demand.
            # Any energy beyond meeting the demand is sold to the grid.
            # --------------------------
            if discharge_possible >= demand:
                sold_energy = discharge_possible - demand
            else:
                sold_energy = 0.0  # Not enough to cover demand, so nothing is sold

            # Compute reward based on the chosen reward function.
            reward = self._compute_reward(demand, sold_energy, price, discharge_possible)
            logger.info(f"Step {self.current_step}: Discharged {discharge_possible:.2f} kWh, Demand={demand:.2f}, Sold Energy={sold_energy:.2f}, Reward={reward:.2f}")


        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Construct the next observation.
        obs = np.array([self.soc, demand, price], dtype=np.float32)

        return obs, reward, done, info

    def _compute_reward(self, demand, sold_energy, price, discharge_amount):
        """
        Compute the reward based on the selected reward function.

        Three reward functions are available:

        1. "profit" (default):
            Reward = \( \text{price} \times \text{sold\_energy} \)

            This function rewards the agent solely based on the revenue generated
            from selling surplus energy to the grid after meeting demand.

        2. "penalty_unmet":
            Reward = \( \text{price} \times \text{discharge\_amount} - \lambda \times \max(0, \text{demand} - \text{discharge\_amount}) \)

            Here, the agent earns revenue for all discharged energy but is penalized
            for any portion of household demand that is not met. This encourages the agent
            to ensure that the internal demand is satisfied before selling energy.

        3. "degradation":
            Reward = \( \text{price} \times \text{sold\_energy} - c \times \text{discharge\_amount} \)

            This reward takes into account battery degradation. While the agent earns revenue
            for selling surplus energy, it incurs a cost proportional to the discharged energy,
            which simulates battery wear-and-tear and encourages cautious discharging.
        """
        if self.reward_type == "profit":
            # --------------------------
            # Reward function 1: Profit Reward (default)
            # --------------------------
            return price * sold_energy

        elif self.reward_type == "penalty_unmet":
            # --------------------------
            # Reward function 2: Penalize Unmet Demand.
            # --------------------------
            penalty = 10.0  # Penalty coefficient (tunable)
            unmet_demand = max(0.0, demand - discharge_amount)
            return price * discharge_amount - penalty * unmet_demand

        elif self.reward_type == "degradation":
            # --------------------------
            # Reward function 3: Battery Degradation Cost Aware.
            # --------------------------
            degradation_cost = 0.5  # Cost per unit of discharged energy (tunable)
            return price * sold_energy - degradation_cost * discharge_amount

        else:
            # Fallback to default profit reward if an unknown reward type is provided.
            return price * sold_energy

    def render(self, mode='human'):
        """
        Render the current state of the environment.
        """
        print(f"Step: {self.current_step}, SoC: {self.soc:.2f}")
