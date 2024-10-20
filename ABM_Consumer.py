import operator
import math
import random

from collections import OrderedDict
from scipy.stats import truncnorm
from mesa import Agent
import numpy as np

from ABM_Smartphones import Smartphone


def distribute_attitude_level(a, b, loc, scale):
    """
    Distribute pro-environmental attitude level toward 
    the decision in the population.
    """
    distribution = truncnorm(a, b, loc, scale)
    attitude_level = float(distribution.rvs(1))
    return attitude_level  
    
    
class Consumer(Agent):
    def __init__(self, unique_id, model, 
                 own_incomes,
                 w_A=0.3, w_SN=0.3, w_PBC=0.4,):
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
        self.EoL_pathway = None
        
        # Get ID for recyclers and refurbisher
        self.recycling_facility_id = model.num_consumers + random.randrange(model.num_recyclers)
        self.refurbisher_id = model.num_consumers + model.num_prod_n_recyc + \
            random.randrange(model.num_refurbishers)
            
        self.landfill_cost = random.choice(landfill_cost)
        self.init_landfill_cost = self.landfill_cost

        # HERE
        self.hoarding_cost = np.random.triangular(
            hoarding_cost[0], hoarding_cost[2], hoarding_cost[1]) * self.max_storage
            
    def get_tpb_attitude(self, decision, att_level_ratios, att_level_reuse, weight_a):
        """
        Calculate pro-environmental attitude component of EoL TPB rule. Options
        considered pro environmental get a higher score than other options.
        """
        if decision == "to_buy":
            return [weight_a * x for x in att_level_ratios]
                    
        all_EoL_pathways = list(self.model.all_EoL_pathways.keys())
        for i in range(len(att_level_ratios)):
            if decision == "EoL_pathway":
                if all_EoL_pathways[i] == "repair" or all_EoL_pathways[i] == "sell" or all_EoL_pathways[i] == "recycle":
                    att_level_ratios[i] = att_level_reuse
                else:
                    att_level_ratios[i] = 1 - att_level_reuse
                    
            elif decision == "purchase_choice":
                if self.purchase_choices[i] == "used":
                    att_level_ratios[i] = att_level_reuse
                else:
                    att_level_ratios[i] = 1 - att_level_reuse
                    
        return [weight_a * x for x in att_level_ratios]
    
    def tpb_perceived_behavioral_control(self, decision, pbc_choice, weight_pbc):
        """
        Calculate perceived behavioral control component of EoL TPB rule.
        behavioral control is understood as a function of financial costs.
        考虑个人收入
        """
        max_cost = max(abs(i) for i in pbc_choice)
        pbc_choice = [i / max_cost for i in pbc_choice]
        if decision == "EoL_pathway":
            self.repairable_modules(pbc_choice)
            if self.model.extended_tpb["Extended tpb"]:
                pbc_choice = \
                    [self.convenience[i] + self.knowledge[i] + pbc_choice[i]
                        for i in range(len(pbc_choice))]
                max_cost = max(abs(i) for i in pbc_choice)
                pbc_choice = [i / max_cost for i in pbc_choice]
        return [weight_pbc * -1 * max(i, 0) for i in pbc_choice]
    
    def tpb_decision(self, decision, list_choices, EoL_pathways, 
                     weight_sn, pbc_choice, weight_pbc, 
                     att_levels_purchase, att_level_reuse, weight_a):
        """
        Select the decision with highest behavioral intention following the
        Theory of `Planned Bahevior` (TPB). `Behavioral intention` is a function
        of the subjective norm, the perceived behavioral control and attitude.
        """
        # Subjective norm (peer pressure)
        sn_values = self.tpb_subjective_norm(
            decision, list_choices, weight_sn)
        # Perceived behavioral control
        pbc_values = self.tpb_perceived_behavioral_control(
            decision, pbc_choice, weight_pbc)
        # Pro-environmental attitude
        a_values = self.tpb_attitude(
            decision=decision, 
            att_levels_purchase=att_levels_purchase, 
            att_level_reuse=att_level_reuse, 
            weight_a=weight_a)
        
        self.behavioral_intentions = [
            (pbc_values[i]) + sn_values[i] + a_values[i] 
                for i in range(len(pbc_values))]
        
        self.pathways_and_BI = {
            list_choices[i]: self.behavioral_intentions[i]
                for i in range(len(list_choices))}
        
        shuffled_dic = list(self.pathways_and_BI.items())
        random.shuffle(shuffled_dic)
        self.pathways_and_BI = OrderedDict(shuffled_dic)
        
        for key, value in self.pathways_and_BI.items():
            if value == np.nan:
                return self.EoL_pathway
            
        conditions = False
        removed_choice = None
        
        while not conditions:
            if removed_choice is not None:
                self.pathways_and_BI.pop(removed_choice)
            if decision == "purchase_choice":
                key = max(self.pathways_and_BI.items(),
                          key=operator.itemgetter(1))[0]
                if self.model.purchase_options.get(key):
                    return key
                else:
                    removed_choice = key
            else:
                key = max(self.pathways_and_BI.items(),
                          key=operator.itemgetter(1))[0]
                if EoL_pathways.get(key) and key != "sell":
                    return key
                else:
                    new_installed_capacity = 0
                    for agent in self.model.schedule.agents:
                        if agent.unique_id < self.model.num_consumers:
                            new_installed_capacity += agent.number_product[-1]
                    used_volume_purchased = self.model.consumer_used_product \
                        / self.model.num_consumers * new_installed_capacity
                if EoL_pathways.get(key) and key == "sell" and \
                        self.sold_waste < used_volume_purchased:
                    return key
                else:
                    removed_choice = key
    
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
        if self.smartphone is None or self.smartphone.status == 0:  # No smartphone or broken
            return True
        # TPB model
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

    def use_smartphone(self):
        """
        Simulate the consumer using the smartphone and updating its state.
        """
        if self.smartphone:
            self.smartphone.update_time_held()

    def decide_end_of_life(self):
        """
        {"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425}
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

    def sell_smartphone(self):
        """
        Simulate the selling of the smartphone.
        """
        if self.smartphone:
            self.smartphone.sell()
            print(f"Consumer {self.consumer_id} sold their smartphone.")
        else:
            print(f"Consumer {self.consumer_id} has no smartphone to sell.")
    
    def recycle_smartphone(self):
        """
        Simulate the recycling of the smartphone.
        """
        if self.smartphone:
            self.smartphone.recycle()
            print(f"Consumer {self.consumer_id} recycled their smartphone.")
        else:
            print(f"Consumer {self.consumer_id} has no smartphone to recycle.")
            
    def landfill_smartphone(self):
        """
        Simulate the landfilling of the smartphone.
        """
        if self.smartphone:
            self.smartphone.landfill()
            print(f"Consumer {self.consumer_id} landfilled their smartphone.")
        else:
            print(f"Consumer {self.consumer_id} has no smartphone to landfill.")
            
    def repair_smartphone(self):
        """
        Simulate the repairing of the smartphone.
        """
        if self.smartphone:
            self.smartphone.repair()
            print(f"Consumer {self.consumer_id} repaired their smartphone.")
        else:
            print(f"Consumer {self.consumer_id} has no smartphone to repair.")
            
    def step(self):
        """
        Main simulation step for the consumer.
        Parameters:
            
        """
        # Step 1: Check if the consumer needs a new smartphone
        to_buy = self.decide_to_purchase_or_not()
        
        if to_buy:
            # Step 2.1: Decide whether to purchase new or used smartphone
            market_choice = self.decide_purchase_new_or_used()
            self.purchase_smartphone(market_choice)
            self.use_smartphone()
            # print(f"Consumer {self.unique_id} decided to purchase a {market_choice} smartphone.")
        else:
            # Step 2.2: Use the smartphone and check EoL decision
            eol_decision = self.decide_end_of_life()
            if eol_decision == "hoard":
                self.use_smartphone()
                
            elif eol_decision == "sell":
                self.sell_smartphone()
                
            elif eol_decision == "recycle":
                self.recycle_smartphone()
                
            elif eol_decision == "landfill":
                self.landfill_smartphone()
                
            elif eol_decision == "repair":
                self.repair_smartphone()
                