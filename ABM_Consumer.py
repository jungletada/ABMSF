import operator
import math
import random

from collections import OrderedDict
from scipy.stats import truncnorm
from mesa import Agent
import numpy as np

from ABM_Smartphones import Smartphone


class Consumer(Agent):
    def __init__(self, unique_id, model,
                 w_A=0.3, w_SN=0.3, w_PBC=0.4,
                 ):
        """
        Initialize a Consumer agent.

        Attributes:
            consumer_id (int): Unique ID for the consumer.
            w_A (float): Weight for attitude in the decision-making model.
            w_SN (float): Weight for subjective norm in the decision-making model.
            w_PBC (float): Weight for perceived behavioral control.
        """
        super().__init__(unique_id, model)
        self.smartphone = None  # Consumer starts with no smartphone

        # To buy or not
        self.w_A_tobuy = 0.45
        self.w_SN_tobuy = 0.20
        self.w_PBC_tobuy = 0.35
           
    def decide_to_purchase_or_not(self, A_tobuy, SN_tobuy, PBC_tobuy, threshold=0.5):
        """
        Decide whether to purchase a smartphone based on TPB model.
        Parameters:
            A_tobuy (float): Attitude towards purchasing a smartphone.
            SN_tobuy (float): Subjective norm (social pressure).
            PBC_tobuy (float): Perceived behavioral control (ease of purchasing).
            threshold (float): Decision threshold for making a purchase.
        Returns:
            bool: True if purchase is made, False otherwise.
        """
        BI_it = self.w_A_tobuy * A_tobuy + self.w_SN_tobuy * SN_tobuy + self.w_PBC_tobuy * PBC_tobuy
        if BI_it > threshold:
            return True  # Consumer decides to purchase
        return False

    def decide_purchase_new_or_used(self, A_used, SN_used, PBC_used, threshold=0.5):
        """
        Decide whether to buy new or used smartphone.
        Parameters:
            A_used (float): Attitude towards purchasing a used smartphone.
            SN_used (float): Subjective norm towards purchasing used.
            PBC_used (float): Perceived behavioral control for purchasing used.
            threshold (float): Decision threshold for making a choice.
        Returns:
            str: "new" or "used" based on the decision.
        """
        BI_used = self.w_A_tobuy * A_used + self.w_A_tobuy * SN_used +self.w_A_tobuy * PBC_used
        if BI_used > threshold:
            return "used"
        return "new"

    def check_smartphone_status(self):
        """
        Check if the consumer needs a new smartphone.
        Returns:
            bool: True if the smartphone is broken or missing, False if still usable.
        """
        if self.smartphone is None or self.smartphone.status == 0:  # No smartphone or broken
            return True
        return False

    def use_smartphone(self):
        """
        Simulate the consumer using the smartphone and updating its state.
        """
        if self.smartphone:
            self.smartphone.update_time_held()

    def decide_end_of_life(self):
        """
        Decide the end-of-life pathway (EoL) for the smartphone (recycle, landfill, etc.).
        Returns:
            str: EoL decision ("recycle", "landfill", "sell").
        """
        # Basic decision process (can be expanded with more detailed conditions)
        if self.smartphone.status == 0:  # Broken smartphone
            return random.choice(["recycle", "landfill", "sell"])  # Random choice for now
        else:
            return "keep using"

    def purchase_smartphone(self, market):
        """
        Simulate the purchase of a smartphone from the market.
        Parameters:
            market (str): Indicates whether to purchase from the "new" or "used" market.
        """
        if market == "new":
            self.smartphone = Smartphone(is_new=True, purchase_price=1000)
        elif market == "used":
            self.smartphone = Smartphone(is_new=False, purchase_price=500, performance=0.8)
        print(f"Consumer {self.consumer_id} purchased a {market} smartphone.")

    def step(self, A_it, SN_it, PBC_it, A_used, SN_used, PBC_used):
        """
        Main simulation step for the consumer.
        Parameters:
            A_it (float): Attitude towards purchasing a smartphone.
            SN_it (float): Subjective norm (social pressure).
            PBC_it (float): Perceived behavioral control (ease of purchasing).
            A_used (float): Attitude towards purchasing a used smartphone.
            SN_used (float): Subjective norm towards purchasing used.
            PBC_used (float): Perceived behavioral control for purchasing used.
        """
        # Step 1: Check if the consumer needs a new smartphone
        if self.check_smartphone_status():
            # Step 2: Decide whether to purchase a smartphone
            if self.decide_to_purchase_or_not(A_it, SN_it, PBC_it):
                # Step 3: Decide whether to purchase new or used
                market_choice = self.decide_purchase_new_or_used(A_used, SN_used, PBC_used)
                self.purchase_smartphone(market_choice)
            else:
                print(f"Consumer {self.consumer_id} decided not to purchase a smartphone.")
        else:
            # Step 4: Use the smartphone and check EoL decision
            self.use_smartphone()
            eol_decision = self.decide_end_of_life()
            if eol_decision != "keep using":
                print(f"Consumer {self.consumer_id} decided to {eol_decision} the smartphone.")
                self.smartphone = None  # After recycling or selling, the smartphone is gone
