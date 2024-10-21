import random
import math

# Newly add:
#   model.num_handled_phones
#   model.initial_phones_handled


class Smartphone:
    """Represents a smartphone in the agent-based model simulation."""

    def __init__(self,
                 is_new,
                 model,
                 performance=1.0,
                 time_held=0,
                 status=1,
                 purchase_price=1000,
                 repair_cost=200,
                 initial_agent_repair_cost=500.0,
                 decay_rate=0.1,
                 user_id=None):
        """
        Initialize a Smartphone instance.

        Attributes:
            is_new (bool): Whether the phone is new (True) or used (False).
            performance (float): Performance level of the phone (0 to 1, where 1 is best performance).
            time_held (int): The time (e.g., months or years) the user has held the phone.
            status (int): If the phone is fine (1) or broken (0).
            purchase_price (float): The price at which the user bought the new phone.
            repair_cost (float): The potential cost to repair the phone if it breaks.
            store_initial_cost (float): Initial repair cost for second-hand store or recycler.
            store_handled_phones (int): Initial number of phones handled by the store or recycler.
            decay_rate (float): The rate at which the performance degrades over time (exponential decay).
            user_id (int/None): ID of the current user who holds the phone (could be a consumer or second-hand store).
        """
        self.model = model
        self.is_new = is_new # True：new smartphone; False: used smartphone
        self.performance = performance  # Range from 0 to 1, where 1 is perfect condition
        self.time_held = time_held  # Number of time units (e.g., months or years) held by the user
        self.status = status  # 1 if functioning, 0 if broken
        self.purchase_price = purchase_price  # Purchase price from the new market
        self.repair_cost = repair_cost  # Repair cost if the phone is broken
        self.initial_agent_repair_cost = initial_agent_repair_cost # Initial repair cost for second-hand store and recycler.
        self.user_id = user_id  # ID of the current user holding this phone
        self.decay_rate = decay_rate # Rate at which the performance degrades (lambda in the exponential decay model)
        
        # Additional attributes
        self.eol_probability = 0.05  # Probability that the phone reaches end-of-life per time step
        self.resell_value = self.calculate_resell_price()  # Value if resold in the second-hand market
        self.warranty_duration = 12 if self.is_new else 0  # New phones come with 12 months warranty

    def degrade_performance(self):
        """
        Simulate the degradation of performance over time using an exponential decay model with Gaussian noise.
        """
        if self.status == 1:  # Degrade only if the phone is functioning
            # Calculate the exponential decay
            decay_factor = math.exp(-self.decay_rate * self.time_held)
            self.performance = max(0, self.performance * decay_factor)

            # Add Gaussian noise to the performance
            noise = random.gauss(0, 0.01)  # Mean of 0, standard deviation of 0.02
            self.performance = max(0, min(1, self.performance + noise))  # Ensure performance stays between 0 and 1

            # Update the status if the phone reaches a critical performance level or by random chance
            if random.random() < self.eol_probability or self.performance <= 0.05:
                self.status = 0  # Phone breaks down if end-of-life probability is reached or performance too low
        else:
            self.performance = 0  # If phone is broken, performance is zero

    def update_time_held(self):
        """Increment the time held by the user."""
        self.time_held += 1
        self.degrade_performance()

    def calculate_repair_cost(self):
        """Attempt to repair the phone, considering warranty status and learning effects.

        If the phone is under warranty (self.time_held < self.warranty_duration), 
        the repair cost is set to a very low value. Otherwise, the repair cost is calculated
        based on phone's performance and the learning effect of the store.
        store_initial_cost (float): Initial repair cost for second-hand store or recycler.
        store_handled_phones (int): current number of phones handled by the store or recycler.
        """
        if self.time_held < self.warranty_duration:  # Phone is under warranty
            self.repair_cost = 0  # Free repair under warranty (or set to a very low cost)
            print(f"Phone is under warranty. Repair cost: {self.repair_cost}")
        
        else:  # Phone is out of warranty, calculate repair cost based on performance and learning effect
            epsilon = 0.5
            cost_max = 0.2 * self.purchase_price  # Set the maximum repair cost (20% of purchase price)
            # Basic repair cost based on performance
            basic_repair_cost = cost_max * (1 - self.performance)
            # Learning effect: Reduce cost based on the number of phones handled by the store
            learning_factor = self.initial_agent_repair_cost * (self.model.num_handled_phones / max(1, self.model.initial_phones_handled))
            self.repair_cost = basic_repair_cost + epsilon * learning_factor
            print(f"Phone is out of warranty. Repair cost with learning effect: {self.repair_cost:.2f}")
            return self.repair_cost
    
    def repair_product(self):
        """Attempt to repair the phone with a random chance of success."""
        # Attempt to repair the phone
        repair_success = random.random()  # Random chance to repair the phone
        if repair_success > 0.9:  # 50% chance of successful repair
            self.status = 1
            self.performance = min(0.9, self.performance + 0.3)  # Performance improves after repair
            return True  # Repair was successful
        return False  # Repair failed

    def calculate_resell_price(self):
        """
        Calculate the resell value of the smartphone based on performance, status, and age.
        """
        depreciation_factor = 0.8 if self.is_new else 0.6
        age_factor = max(0, 1 - 0.05 * self.time_held)  # The longer the time held, the lower the value
        resell_price = self.purchase_price * depreciation_factor * self.performance * age_factor
        return resell_price
    
    def calculate_recycle_price(self):
        """
        """
        recycle_price = 100
        return recycle_price

    def recycle_product(self, new_owner_id):
        """
        """
        self.user_id = new_owner_id  # Update the user ID to the new owner

    def resell_product(self, new_owner_id):
        """
        Simulate selling the phone to a new user or a second-hand store.
        Parameters:
            new_owner_id (int/None): The ID of the new user/store that buys the phone.
        Returns:
            float: The resell value of the phone.
        """
        self.user_id = new_owner_id  # Update the user ID to the new owner

    def __repr__(self):
        return (f"Smartphone(is_new={self.is_new}, "
                f"performance={self.performance:.2f}, " 
                f"time_held={self.time_held}, "
                f"status={'working' if self.status == 1 else 'broken'}, "
                f"purchase_price={self.purchase_price}, "
                f"user_id={self.user_id}, "
                f"resell_value={self.resell_value:.2f})")
