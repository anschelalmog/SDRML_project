# environment.py
import gym
import numpy as np
from gym import spaces

class BatteryEnv(gym.Env):
    """
    A custom environment modeling an electricity market with:
      - State of Charge (SoC) of the battery
      - Demand (household load)
      - Price (for selling to the grid)

    Continuous action space: charging/discharging the battery in range [-capacity, +capacity].
    """
    def __init__(self, config):
        super(BatteryEnv, self).__init__()

        self.config = config
        self.capacity = config["battery_capacity"]
        self.max_steps = config["max_steps"]

        # For simplicity, let's treat each "episode step" as one "time unit."
        # We'll define a "day length" to simulate periodic demand/price.
        self.day_length = config["day_length"]
        self.current_step = 0

        # Observation space: (SoC, Demand, Price)
        #   SoC in [0, capacity]
        #   Demand >= 0 (upper bound set large, say 1e9, to avoid clipping)
        #   Price >= 0
        low_obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([self.capacity, 1e9, 1e9], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action space: continuous in [-capacity, +capacity]
        self.action_space = spaces.Box(
            low=np.float32(-self.capacity),
            high=np.float32(self.capacity),
            shape=(1,),
            dtype=np.float32
        )

        self.soc = 0.0  # State of charge
        self.demand = 0.0
        self.price = 0.0

    def reset(self):
        """
        Reset environment state for a new episode.
        """
        self.current_step = 0
        # Start battery half-full for example
        self.soc = self.capacity / 2.0

        # Generate the initial demand and price
        self.demand = self._get_demand(self.current_step)
        self.price = self._get_price(self.current_step)

        return self._get_obs()

    def step(self, action):
        """
        Apply the action (charge or discharge), update SoC, compute reward.
        """
        # Clamp the action to the valid range
        action = np.clip(action[0], -self.capacity, self.capacity)

        # If action > 0 => charging, if action < 0 => discharging
        old_soc = self.soc

        # Update SoC with charging or discharging
        new_soc = self.soc + action
        new_soc = np.clip(new_soc, 0.0, self.capacity)
        self.soc = new_soc

        # Next step's demand and price
        self.current_step += 1
        self.demand = self._get_demand(self.current_step)
        self.price = self._get_price(self.current_step)

        # Compute how much energy we *actually discharged*
        #   action < 0 => discharge, action > 0 => charge
        actual_discharge = max(0.0, old_soc - self.soc)  # how much SoC actually went out

        # Out of that discharged energy, some portion meets the internal demand
        energy_used_for_demand = min(self.demand, actual_discharge)
        leftover = actual_discharge - energy_used_for_demand

        # The agent sells "leftover" to the grid at the current price
        reward = leftover * self.price

        # Observe, check if done
        done = (self.current_step >= self.max_steps)

        info = {}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.array([self.soc, self.demand, self.price], dtype=np.float32)

    def _get_demand(self, t):
        """
        Demand is a function of time with two peaks + random noise.
        Example formula from the project instructions:
           f(x) = 100 * exp(-((x-0.4)^2)/(2*0.05^2)) +
                  120 * exp(-((x-0.7)^2)/(2*0.1^2))
        We'll treat x as fraction of the day. Then add noise.
        """
        x = (t % self.day_length) / self.day_length
        val = (100 * np.exp(-((x-0.4)**2)/(2*(0.05**2))) +
               120 * np.exp(-((x-0.7)**2)/(2*(0.1**2))))
        # Add random noise
        noise = np.random.normal(0, self.config["demand_noise_std"])
        return max(0.0, val + noise)

    def _get_price(self, t):
        """
        Similar approach for price. For variety, let's make it smaller amplitude.
        """
        x = (t % self.day_length) / self.day_length
        val = (50 * np.exp(-((x-0.3)**2)/(2*(0.07**2))) +
               70 * np.exp(-((x-0.65)**2)/(2*(0.1**2))))
        noise = np.random.normal(0, self.config["price_noise_std"])
        return max(0.0, val + noise)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
