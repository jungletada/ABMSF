import random

class Smartphone:
    def __init__(
        self, 
        is_new=True, 
        performance=1.0, 
        time_held=0, 
        status=1, 
        purchase_price=0, 
        repair_cost=0, 
        consumer_id=None):
        """
        Initialize a Smartphone instance.

        Attributes:
            is_new (bool): Whether the phone is new (True) or used (False).
            performance (float): Performance level of the phone (0 to 1, where 1 is best performance).
            time_held (int): The time (e.g., months or years) the consumer has held the phone.
            status (int): If the phone is fine (1) or broken (0).
            purchase_price (float): The price at which the consumer bought the phone.
            repair_cost (float): The potential cost to repair the phone if it breaks.
            consumer_id (int/None): ID of the consumer who holds the phone (if any).
        """
        self.is_new = is_new
        self.performance = performance  # Range from 0 to 1, where 1 is perfect condition
        self.time_held = time_held  # Number of time units (e.g., months or years) held by consumer
        self.status = status  # 1 if functioning, 0 if broken
        self.purchase_price = purchase_price  # Purchase price from either new or second-hand market
        self.repair_cost = repair_cost  # Repair cost if the phone is broken
        self.consumer_id = consumer_id  # ID of the consumer holding this phone
        
        # Additional attributes
        self.eol_probability = 0.05  # Probability that the phone reaches end-of-life per time step
        self.resell_value = self.calculate_resell_value()  # Value if resold in the second-hand market
        self.warranty_duration = 12 if self.is_new else 0  # New phones come with 12 months warranty
        self.battery_life = random.uniform(0.8, 1.0) if self.is_new else random.uniform(0.4, 0.9)  # Battery life quality
        
    def calculate_resell_value(self):
        """Calculate the resell value of the smartphone based on performance, status, and age."""
        depreciation_factor = 0.8 if self.is_new else 0.6
        age_factor = max(0, 1 - 0.05 * self.time_held)  # The longer the time held, the lower the value
        resell_value = self.purchase_price * depreciation_factor * self.performance * age_factor
        return resell_value

    def degrade_performance(self):
        """Simulate the degradation of performance over time."""
        if self.status == 1:  # Degrade only if the phone is functioning
            degradation_rate = 0.02  # Performance degrades by 2% per time unit
            self.performance = max(0, self.performance - degradation_rate)
            if random.random() < self.eol_probability:
                self.status = 0  # Phone breaks down if end-of-life probability is reached
        else:
            self.performance = 0  # If phone is broken, performance is zero

    def repair(self):
        """Attempt to repair the phone."""
        if self.status == 0:
            repair_success = random.random()  # Random chance to repair the phone
            if repair_success > 0.5:  # 50% chance to successfully repair the phone
                self.status = 1
                self.performance = min(0.9, self.performance + 0.3)  # After repair, performance improves slightly
                return True  # Repair was successful
            else:
                return False  # Repair failed
        return False  # No need for repair if phone is not broken

    def update_time_held(self):
        """Increment the time held by the consumer."""
        self.time_held += 1
        self.degrade_performance()

    def sell(self):
        """Simulate selling the phone in the second-hand market."""
        if self.status == 1:
            return self.resell_value  # Sell the phone at its calculated resell value
        else:
            return 0  # Can't sell a broken phone

    def __repr__(self):
        return (f"Smartphone(is_new={self.is_new}, performance={self.performance:.2f}, time_held={self.time_held}, "
                f"status={'working' if self.status == 1 else 'broken'}, "
                f"purchase_price={self.purchase_price}, "
                f"resell_value={self.resell_value:.2f})")
