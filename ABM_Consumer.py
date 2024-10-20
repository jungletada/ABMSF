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
                 w_A=0.3, w_SN=0.3, w_PBC=0.4,
                 landfill_cost=100,
                 hoarding_cost=[2,1,2]):
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
        self.A_tobuy = 1
        self.w_A_buy_or_not = 0.45
        self.w_SN_buy_or_not = 0.20
        self.w_PBC_buy_or_not = 1 - self.w_A_buy_or_not - self.w_SN_buy_or_not
        
        self.EoL_pathway = None
        self.to_buy = False
        
        # Get ID for recyclers and refurbisher
        self.recycling_facility_id = model.num_consumers + random.randrange(model.num_recyclers)
        self.refurbisher_id = model.num_consumers + model.num_prod_n_recyc + \
            random.randrange(model.num_refurbishers)
            
        self.landfill_cost = random.choice(landfill_cost)
        self.init_landfill_cost = self.landfill_cost

        # HERE
        self.hoarding_cost = np.random.triangular(
            hoarding_cost[0], hoarding_cost[2], hoarding_cost[1]) * self.max_storage
            
    def tpb_attitude(self, decision, att_level_reuse, weight_att):
        """
        Calculate pro-environmental attitude component of EoL TPB rule. Options
        considered pro environmental get a higher score than other options.
        Parameters:
            decision (str): The type of decision being made (e.g., "EoL_pathway" or "purchase_choice").
            att_level_reuse (float): The attitude level towards reuse/pro-environmental options.
            weight_att (dict): A dictionary of weights for each option in the attitude calculation.
        Returns:
            att_level_ratios (dict): A dictionary containing the calculated attitude levels for each option,
                                     weighted by the corresponding attitude weight.
        Return
        """
        att_level_ratios = {}
        all_EoL_pathways = list(self.model.all_EoL_pathways.keys())
        for i, pathway in enumerate(all_EoL_pathways):
            if decision == "EoL_pathway":
                if pathway in ["repair", "sell", "recycle"]:
                    att_level_ratios[pathway] = att_level_reuse * weight_att[pathway]
                else:
                    att_level_ratios[pathway] = (1 - att_level_reuse) * att_level_ratios
            
            elif decision == "purchase_choice":
                if self.purchase_choices[i] == "used":
                    att_level_ratios[pathway] = att_level_reuse * weight_att[pathway]
                else:
                    att_level_ratios[pathway] = (1 - att_level_reuse) * weight_att[pathway]
        
        return att_level_ratios
    
    def tpb_subjective_norm(self, decision, list_choices, weight_sn):
        """
        Parameters:
            decision (str): The type of decision being made (e.g., "repair, sell, recycle" or "used, new").
            list_choices (list): A list of possible choices for the decision.
            weight_sn (dict): A dictionary of weights for each choice in the subjective norm calculation.
        Returns:
            proportion_sn (dict): A dictionary containing the proportion of neighbors making each choice, 
                weighted by the corresponding subjective norm weight.
        Calculate subjective norm (peer pressure) component of TPB rule
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbor_agents = list(self.model.grid.get_cell_list_contents(neighbors_nodes))
        proportion_sn = {}
        
        for choice in list_choices:
            proportion_choice = sum(1 for agent in neighbor_agents if getattr(agent, decision) == choice) / len(neighbor_agents)
            proportion_sn[choice] = proportion_choice * weight_sn[choice]

        return proportion_sn
    
    def tpb_perceived_behavioral_control(self, decision, pbc_costs, weight_pbc):
        """
        Parameters:
            decision (str): The type of decision being made (e.g., "EoL_pathway" or "purchase_choice").
            pbc_costs (dict): A dictionary containing the costs associated with each option.
            weight_pbc (dict): A dictionary of weights for each option in the perceived behavioral control calculation.
        
        Returns:
            dict: A dictionary containing the calculated perceived behavioral control values for each option,
                  normalized and weighted by the corresponding PBC weight.
        
        This function calculates the perceived behavioral control (PBC) component of the Theory of Planned Behavior (TPB) rule.
        For purchase decisions, it considers only the financial costs. For End-of-Life (EoL) pathway decisions, it may also
        include factors like convenience and knowledge if the extended TPB model is enabled.
        """
        # PBC for purchasing smartphones
        max_cost = max(abs(i) for i in pbc_costs.values())
        if decision == "purchase_choice":
            return {key: -1 * value / max_cost * weight_pbc[key] for key, value in pbc_costs.items()}
        
        # PBC for EoL pathways
        elif decision == "EoL_pathway":
            pbc_eol = {}
            self.repairable_modules(pbc_costs)
            if self.model.extended_tpb["Extended tpb"]:
                for key in pbc_costs.keys():
                    pbc_eol[key] = self.convenience[key] + self.knowledge[key] + (pbc_costs[key] / max_cost)
            max_eol = max(abs(i) for i in pbc_eol.values())
            return {key: -1 * value / max_eol * weight_pbc[key] for key, value in pbc_eol.items()}
    
    def tpb_decision(self, decision, list_choices, EoL_pathways, 
                     weight_sn, pbc_choice, weight_pbc, 
                     att_levels_purchase, att_level_reuse, weight_a):
        """
        Select the decision with highest behavioral intention following the
        Theory of `Planned Bahevior` (TPB). `Behavioral intention` is a function
        of the subjective norm, the perceived behavioral control and attitude.
        """
        # Subjective norm (peer pressure)
        sn_values = self.tpb_subjective_norm(decision, list_choices, weight_sn)
        # Perceived behavioral control
        pbc_values = self.tpb_perceived_behavioral_control(decision, pbc_choice, weight_pbc)
        # Pro-environmental attitude
        a_values = self.tpb_attitude(
            decision=decision,
            att_level_ratios=att_levels_purchase,
            att_level_reuse=att_level_reuse,
            weight_att=weight_a)
        
        self.behavioral_intentions = [
            (pbc_values[i]) + sn_values[i] + a_values[i]
                for i in range(len(pbc_values))]
        
        self.pathways_and_BI = {
            list_choices[i]: self.behavioral_intentions[i]
                for i in range(len(list_choices))}

    def attitude_to_buy(self):
        """Depends on the incomes of the consumer"""
        
        pass
        
    def decide_to_purchase_or_not(self):
        """
        Decide whether to purchase a smartphone based on TPB model.
        Parameters:
            
        Returns:
            bool: True if purchase is made, False otherwise.
        """
        if self.smartphone is None or self.smartphone.status == 0:  # No smartphone or broken
            return True
        # TPB model
        A_tobuy = self.attitude_to_buy()
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        num_all_neighbors = len(list(self.model.grid.get_cell_list_contents(neighbors_nodes)))
        proportion_tobuy = len(
            [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
                if getattr(agent, 'to_buy')]) / num_all_neighbors
        PBC_tobuy =  - (self.model.avg_price / self.model.highest_price)
        
        BI_tobuy = self.w_A_buy_or_not * A_tobuy + self.w_SN_buy_or_not * proportion_tobuy + self.w_PBC_buy_or_not * PBC_tobuy
        BI_notbuy = self.w_A_buy_or_not * (1 -  A_tobuy) + self.w_SN_buy_or_not * (1 - proportion_tobuy) # PBC_notbuy = 0
        
        if BI_tobuy > BI_notbuy:
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
        BI_used = self.w_A_buy_or_not * A_used + self.w_A_buy_or_not * SN_used +self.w_A_buy_or_not * PBC_used
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
                