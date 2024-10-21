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
                 max_time_hoard=12 * 5,
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
        self.decision = None
        self.EoL_pathway = None        
        self.pathway_choice = None
        self.to_purchase = False
        
        self.w_att_buy_or_not = 0.45
        self.w_sn_buy_or_not = 0.20
        self.w_pbc_buy_or_not = 1 - self.w_att_buy_or_not - self.w_sn_buy_or_not
        
        self.income = self.model.consumer_incomes[unique_id]
        
        self.repair_cost = 0
        self.sell_cost = 0
        self.landfill_cost = 0
        self.hoard_cost = 0
        self.recycle_cost = 0
        self.max_time_hoard = max_time_hoard
        self.num_cumulative_purchase_new = 0
        self.num_cumulative_purchase_used = 0
        
        # column sum up to 1
        self.weight_att_eol = {"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425} 
        self.weight_sn_eol = {"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425} 
        self.weight_pbc_eol = {"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425} 
        
        # column sum up to 1
        self.weight_att_purchase = {'used': 0.05, 'new': 0.95}
        self.weight_sn_purchase = {'used': 0.05, 'new': 0.95}
        self.weight_pbc_purchase = {'used': 0.05, 'new': 0.95}
        
        self.eol_pathway_choices = self.model.init_eol_rate.keys()
        self.purchase_choices = self.model.init_purchase_rate.keys()
        
        self.pbc_costs_purchase = {'used': 2000, 'new': 4000}
        self.pbc_costs_eol = {"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425} 

        self.behavior_intention = {}
        
        # Get ID for recyclers and manufacturer
        self.recycling_facility_id = model.num_consumers + random.randrange(model.num_recyclers)
        self.refurbisher_id = model.num_consumers + model.num_prod_n_recyc + \
            random.randrange(model.num_refurbishers)

    def get_attitude_purchase(self):
        """ Consumer Agent
        Calculate attitudes based on income distribution.
        Attitudes are calculated as a weighted sum of the log-normal distribution.

        Parameters:
            incomes (np.ndarray): Array of accumulated incomes for the population.

        Returns:
            np.ndarray: Array of calculated attitudes based on the income distribution.
        """
        income_min = np.min(self.model.consumer_incomes)
        income_max = np.max(self.model.consumer_incomes)
        attitudes = (np.log(self.income) - np.log(income_min)) / (
            np.log(income_max) - np.log(income_min))
        return attitudes
    
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
        """
        att_level_ratios = {}
        for i, pathway in enumerate(self.eol_pathway_choices):
            if decision == "EoL_pathway":
                if pathway in ["repair", "sell", "recycle"]:
                    att_level_ratios[pathway] = att_level_reuse * weight_att[pathway]
                else:
                    att_level_ratios[pathway] = (1 - att_level_reuse) * weight_att[pathway]
            
            elif decision == "purchase_choice":
                if self.purchase_choices[i] == "used":
                    att_level_ratios[pathway] = att_level_reuse * weight_att[pathway]
                else:
                    att_level_ratios[pathway] = (1 - att_level_reuse) * weight_att[pathway]
        
        return att_level_ratios
    
    def tpb_subjective_norm(self, decision, weight_sn):
        """
        Parameters:
            decision (str): The type of decision being made (e.g., "repair, sell, recycle" or "used, new").
            weight_sn (dict): A dictionary of weights for each choice in the subjective norm calculation.
        Returns:
            proportion_sn (dict): A dictionary containing the proportion of neighbors making each choice, 
                weighted by the corresponding subjective norm weight.
        Calculate subjective norm (peer pressure) component of TPB rule
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbor_agents = list(self.model.grid.get_cell_list_contents(neighbors_nodes))
        proportion_sn = {}
        
        if decision == "EoL_pathway":
            list_choices = self.eol_pathway_choices
        else:
            list_choices = self.purchase_choices
            
        for choice in list_choices:
            proportion_choice = sum(1 for agent in neighbor_agents 
                                        if getattr(agent, 'pathway_choice') == choice) / len(neighbor_agents)
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
    
    def tpb_decision(self, 
                     decision, 
                     weight_att, 
                     weight_sn, 
                     weight_pbc, 
                     pbc_costs, 
                     att_level_reuse):
        """
        Select the decision with highest behavioral intention following the
        Theory of `Planned Bahevior` (TPB). `Behavioral intention` is a function
        of the subjective norm, the perceived behavioral control and attitude.
        Parameters:
            decision (str): The type of decision being made (e.g., "EoL_pathway" or "purchase_choice").
            pbc_costs (dict): A dictionary containing the costs associated with each option.
        """
        # Subjective norm (peer pressure)
        sn_values = self.tpb_subjective_norm(decision, weight_sn)
        # Perceived behavioral control
        pbc_values = self.tpb_perceived_behavioral_control(decision, pbc_costs, weight_pbc)
        # Pro-environmental attitude
        att_values = self.tpb_attitude(
            decision=decision,
            att_level_reuse=att_level_reuse,
            weight_att=weight_att)
        
        if decision == "EoL_pathway":
            list_choices = self.eol_pathway_choices
        else:
            list_choices = self.purchase_choices
            
        self.behavior_intention = {}
        for choice in list_choices:
            self.behavior_intention[choice] = pbc_values[choice] + sn_values[choice] + att_values[choice]

        self.pathway_choice = max(self.behavior_intention, key=self.behavior_intention.get)
        if self.pathway_choice == "hoard" and self.smartphone.time_held >= self.max_time_hoard: # exceed max_time_hoard
            # Choose the second biggest value in 'self.behavior_intention'
            sorted_intentions = sorted(self.behavior_intention.items(), key=lambda x: x[1], reverse=True)
            self.pathway_choice = sorted_intentions[1][0]  # Get the key of the second highest value
            
        print(f'Consumer {self.unique_id} decides to {self.pathway_choice}.')
  
    def decide_to_purchase_or_not(self):
        """
        Decide whether to purchase a smartphone based on TPB model.
        Parameters:
            
        Returns:
            bool: True if purchase is made, False otherwise.
        """
        if self.smartphone is None or self.smartphone.status == 0:  # No smartphone
            self.to_purchase = True
        # TPB model
        att_tobuy = self.get_attitude_purchase()
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        num_all_neighbors = len(list(self.model.grid.get_cell_list_contents(neighbors_nodes)))
        proportion_tobuy = len(
            [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
                if getattr(agent, 'to_purchase')]) / num_all_neighbors
        pbc_tobuy =  - (self.model.avg_price / self.model.highest_price)

        behavior_intention_tobuy = self.w_att_buy_or_not * att_tobuy \
                                + self.w_sn_buy_or_not * proportion_tobuy \
                                + self.w_pbc_buy_or_not * pbc_tobuy

        pbc_notbuy = -self.repair_cost / self.income
        behavior_intention_notbuy = self.w_att_buy_or_not * (1 - att_tobuy) \
                                + self.w_sn_buy_or_not * (1 - proportion_tobuy) \
                                + self.w_pbc_buy_or_not * pbc_notbuy
        
        self.to_purchase = behavior_intention_tobuy >= behavior_intention_notbuy

    def use_smartphone(self):
        """
        Simulate the consumer using the smartphone and updating its state.
        """
        if self.smartphone:
            self.smartphone.update_time_held()

    def get_eol_cost(self):
        """Calculate and set the costs for different end-of-life options."""
        self.repair_cost = self.smartphone.calculate_repair_cost() # repair cost need to be paid by consumer
        self.sell_cost = -self.smartphone.calculate_resell_price() # resell cost is paid by second-hand store
        self.recycle_cost = -self.smartphone.calculate_repair_cost() # recycle cost is paid by second-hand store
        self.landfill_cost = 0
        self.hoard_cost = 0

        self.pbc_costs_eol = {
            "repair": self.repair_cost / self.smartphone.purchase_price, 
            "sell": self.sell_cost / self.smartphone.purchase_price, 
            "recycle": self.recycle_cost / self.smartphone.purchase_price, 
            "landfill": 0, 
            "hoard": 0} 

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

    def sell_smartphone(self, new_owner_id):
        """
        Simulate the selling of the smartphone.
        """
        self.smartphone.resell_product(new_owner_id)
        # 更改产品的所有权
        self.smartphone = None
        print(f"Consumer {self.consumer_id} sold their smartphone.")
    
    def recycle_smartphone(self, new_owner_id):
        """
        Simulate the recycling of the smartphone.
        """
        self.smartphone.recycle_product(new_owner_id)
        # 更改产品的所有权
        self.smartphone = None
        print(f"Consumer {self.consumer_id} recycled their smartphone.")
            
    def landfill_smartphone(self):
        """
        Simulate the landfilling of the smartphone.
        """
        self.smartphone = None
        print(f"Consumer {self.consumer_id} landfilled their smartphone.")
            
    def repair_smartphone(self):
        """
        Simulate the repairing of the smartphone.
        """
        self.smartphone.repair_product()
        print(f"Consumer {self.consumer_id} repaired their smartphone.")
         
    def step(self):
        """
        Main simulation step for the consumer.
        """
        # Step 1: Check if the consumer needs a new smartphone and update 'self.to_purchase'
        self.decide_to_purchase_or_not()
        
        if self.to_purchase:
            # Step 2.1: Decide whether to purchase new or used smartphone
            # update 'self.pathway_choice'
            self.tpb_decision(decision='purchase_choice',
                              weight_att=self.weight_att_purchase,
                              weight_sn=self.weight_sn_purchase,
                              weight_pbc=self.weight_pbc_purchase,
                              pbc_costs=self.pbc_costs_purchase,
                              att_level_reuse=0.7)
            self.purchase_smartphone(self.pathway_choice)
            if self.pathway_choice == 'used':
                self.num_cumulative_purchase_used += 1
            else:
                self.num_cumulative_purchase_new += 1
            self.use_smartphone()
        else:
            # Step 2.2: Use the smartphone and check EoL decision
            # update 'self.pathway_choice'
            self.tpb_decision(decision='EoL_pathway',
                              weight_att=self.weight_att_eol,
                              weight_sn=self.weight_sn_eol,
                              weight_pbc=self.weight_pbc_eol,
                              pbc_costs=self.pbc_costs_eol,
                              att_level_reuse=0.2)
            
            if self.pathway_choice == "hoard":
                self.use_smartphone() # update the time held and performance

            elif self.pathway_choice == "sell":
                self.sell_smartphone()

            elif self.pathway_choice == "recycle":
                self.recycle_smartphone()

            elif self.pathway_choice == "landfill":
                self.landfill_smartphone()
                
            elif self.pathway_choice == "repair":
                self.repair_smartphone()
                
