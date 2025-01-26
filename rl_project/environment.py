import gym
import numpy as np
from gym import spaces


class ElectricityMarketEnv(gym.Env):
    """
    A custom Environment for an Electricity Market RL problem with battery storage.
    -------------------------------------------------------------------------------
    STATE:
        [SoC, Demand, Price]
        - SoC: State of Charge of the battery (0 <= SoC <= battery_capacity)
        - Demand: The electricity demand at the current timestep
        - Price: The electricity price at the current timestep

    ACTION (continuous):
        action in [-battery_capacity, +battery_capacity]
        - Negative values (<=0) mean "charge" the battery.
        - Positive values (>=0) mean "discharge" the battery.

    REWARD:
        Three possible reward modes are supported (choose with 'reward_mode'):
          1) 'profit_only'
             reward = (energy_sold_to_grid) * (price)
               where energy_sold_to_grid = max(0, discharge_after_meeting_demand)

          2) 'profit_with_unmet_penalty'
             reward = (energy_sold_to_grid) * (price) - penalty_unmet * (unmet_demand > 0)
               i.e., if demand couldn't be met from SoC, penalize.

          3) 'profit_minus_battery_degradation'
             reward = (energy_sold_to_grid) * (price) - alpha * (abs(charge_action))
               i.e., we penalize the battery usage to reflect degradation costs
               (both for charging and discharging).

    EPISODE:
      - Single episode runs for self.max_steps steps or until done.
      - We add a time index that increments at each step.
    """

    def __init__(self, args):
        """
        Args:
            battery_capacity (float): Maximum battery storage capacity.
            max_steps (int): Number of steps per episode.
            reward_mode (str): Choose from ['profit_only',
                                           'profit_with_unmet_penalty',
                                           'profit_minus_battery_degradation'].
            penalty_unmet (float): Penalty value for unmet demand (used in 'profit_with_unmet_penalty').
            alpha (float): Battery usage cost coefficient (used in 'profit_minus_battery_degradation').
            seed (int): Random seed.
        """
        super(ElectricityMarketEnv, self).__init__()

        # Environment parameters
        self.battery_capacity = args.get('battery_capacity', 100.0)
        self.max_steps = args.get('max_steps', 100)
        self.reward_mode = args.get('reward_mode', 'profit_only')
        self.penalty_unmet = args.get('penalty_unmet', 50.0)
        self.alpha = args.get('alpha', 0.05)
        self.seed(args.get('seed', 42))

        # continuous 1D action space in [-battery_capacity, +battery_capacity]
        self.action_space = spaces.Box(low=-self.battery_capacity,
                                       high=self.battery_capacity,
                                       shape=(1,), dtype=np.float32)

        # continuous 3D state space: [SoC in [0, battery_capacity], Demand >= 0, Price >= 0]
        # We'll put an upper bound on Demand and Price for safety, though in principle they can be large.
        self.state_space = spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                      high=np.array([self.battery_capacity, 1e6, 1e6], dtype=np.float32),
                                      dtype=np.float32)

        # Internal variables
        self.soc, self.demand, self.price, self.current_step = None, None, None, 0

        # Pre-generate demand & price for the entire episode (or regenerate on each reset)
        self.demand_profile, self.price_profile = None, None

    def seed(self, seed=None):
        """Sets the random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _generate_demand(self, t_normalized):
        """
        Periodic demand function with two peaks + noise.
        For example: f(t) = 100 * exp(-(t - 0.4)^2 / (2 * 0.05^2))
                       + 120 * exp(-(t - 0.7)^2 / (2 * 0.1^2))
                     + some random noise
        """
        val1 = 100 * np.exp(-((t_normalized - 0.4) ** 2) / (2 * (0.05 ** 2)))
        val2 = 120 * np.exp(-((t_normalized - 0.7) ** 2) / (2 * (0.1 ** 2)))
        noise = self.np_random.normal(0, 3)  # small noise
        return max(0.0, val1 + val2 + noise)

    def _generate_price(self, t_normalized):
        """
        Periodic price function with two peaks + noise.
        For example, we can shift amplitudes or frequencies to represent price changes over the day.
        """
        # We'll do something slightly simpler but similarly shaped.
        val1 = 50 * np.exp(-((t_normalized - 0.3) ** 2) / (2 * (0.06 ** 2)))
        val2 = 70 * np.exp(-((t_normalized - 0.8) ** 2) / (2 * (0.08 ** 2)))
        noise = self.np_random.normal(0, 2)  # small noise
        return max(0.0, val1 + val2 + noise)

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        self.current_step = 0

        # Generate new demand & price profiles for the entire episode
        time_indices = np.linspace(0, 1, self.max_steps)
        self.demand_profile = [self._generate_demand(tn) for tn in time_indices]
        self.price_profile = [self._generate_price(tn) for tn in time_indices]

        # Initialize battery SoC at half capacity (arbitrary choice)
        self.soc = self.battery_capacity / 2.0

        # Get initial demand and price
        self.demand = self.demand_profile[self.current_step]
        self.price = self.price_profile[self.current_step]

        return self._get_observation()

    def _get_observation(self):
        return np.array([self.soc, self.demand, self.price], dtype=np.float32)

    def step(self, action):
        """
        Executes the action and updates the environment state.
        Args:
            action (float): The chosen action in [-battery_capacity, battery_capacity].
                            Positive => discharge, Negative => charge.
        Returns:
            observation (np.array): The next observation [SoC, demand, price].
            reward (float): The immediate reward.
            done (bool): Whether the episode has ended.
            info (dict): Extra diagnostic information.
        """
        action_value = np.clip(action, -self.battery_capacity, self.battery_capacity)

        # Battery update
        # If action_value > 0 => Discharge
        # If action_value < 0 => Charge
        # Ensure SoC remains in [0, battery_capacity]

        old_soc = self.soc
        if action_value >= 0:
            # Discharging
            discharge_amount = min(action_value, self.soc)
            self.soc -= discharge_amount
        else:
            # Charging
            charge_amount = min(-action_value, self.battery_capacity - self.soc)
            self.soc += charge_amount

        # Now compute how much power is used to meet demand
        # If we are discharging, we meet demand first from 'discharge_amount'
        # leftover = discharge_amount - demand
        # if leftover > 0 => sold to grid
        # if leftover < 0 => unmet demand (if the environment requires penalizing that)

        if action_value > 0:
            # Discharging portion
            discharge_amount = old_soc - self.soc  # how much we actually discharged
            if discharge_amount >= self.demand:
                # Demand is fully met
                leftover = discharge_amount - self.demand
                unmet_demand = 0.0
            else:
                # Demand is not fully met
                leftover = 0.0
                unmet_demand = self.demand - discharge_amount
        else:
            # If we are charging or action_value <= 0, we do not discharge for demand
            # So if there's demand, it's not met from the battery
            leftover = 0.0
            unmet_demand = self.demand

        # Calculate reward based on the chosen reward mode
        reward = self._calculate_reward(leftover, unmet_demand, action_value)

        # Step forward in time
        self.current_step += 1

        # Check if we reached the end of the episode
        done = self.current_step >= self.max_steps

        # Update next state if not done
        if not done:
            self.demand = self.demand_profile[self.current_step]
            self.price = self.price_profile[self.current_step]

        return self._get_observation(), reward, done, {}

    def _calculate_reward(self, leftover, unmet_demand, action_value):
        """
        Computes the reward depending on the selected reward mode.
        """
        energy_sold_to_grid = leftover
        price = self.price

        if self.reward_mode == 'profit_only':
            # Reward is simply leftover * price
            reward = energy_sold_to_grid * price

        elif self.reward_mode == 'profit_with_unmet_penalty':
            # Subtract penalty if there's any unmet demand
            # (If unmet_demand > 0, we apply a penalty once, ignoring the exact amount.)
            penalty = self.penalty_unmet if unmet_demand > 0 else 0.0
            reward = (energy_sold_to_grid * price) - penalty

        elif self.reward_mode == 'profit_minus_battery_degradation':
            # We penalize battery usage
            usage_cost = self.alpha * abs(action_value)
            reward = (energy_sold_to_grid * price) - usage_cost

        else:
            raise ValueError(f"Invalid reward_mode selected: {self.reward_mode}")

        return reward

    def render(self, mode='human'):
        """Optional: Provide visualization if needed."""
        print(f"Step: {self.current_step}, SoC: {self.soc:.2f}, Demand: {self.demand:.2f}, Price: {self.price:.2f}")
