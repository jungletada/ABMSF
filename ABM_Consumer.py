import operator
import math
import random
import numpy as np

from collections import OrderedDict
from scipy.stats import truncnorm
from mesa import Agent

from ABM_Manufacturer import Manufacturer
from ABM_SecHandStore import SecondHandStore
from ABM_Recycler import Recycler


def distribute_attitude_level(a, b, loc, scale):
    """
    Distribute pro-environmental attitude level toward 
    the decision in the population.
    """
    distribution = truncnorm(a, b, loc, scale)
    attitude_level = float(distribution.rvs(1))
    return attitude_level 


class Consumer(Agent):
    """
    A class representing a consumer agent in the smartphone market simulation.

    This class models the behavior of individual consumers, including their
    decision-making processes for purchasing smartphones (new or used) and
    choosing end-of-life (EoL) pathways for their devices. The Consumer class
    incorporates various factors such as attitudes, subjective norms, and
    perceived behavioral control, which influence the agent's decisions.

    Key features:
    - Smartphone ownership and management
    - Decision-making for purchasing and EoL choices
    - Income-based behavior
    - Tracking of cumulative purchases (new and used)
    - Customizable weights for different decision factors
    - Costs associated with various EoL pathways
    """
    def __init__(
            self,
            model,
            unique_id,
            init_purchase_dist=0.35,
            max_time_hoard=60):
        """
        Initialize a Consumer agent.

        Attributes:
            consumer_id (int): Unique ID for the consumer.
            w_A (float): Weight for attitude in the decision-making model.
            w_SN (float): Weight for subjective norm in the decision-making model.
            w_PBC (float): Weight for perceived behavioral control.
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.smartphone = None  # Consumer starts with no smartphone
        # To buy or not
        self.decision = None
        self.eol_pathway = None
        self.to_purchase = True
        self.pathway_choice = 'new'
        # 'used' if random.random() <= init_purchase_dist else 'new'
        
        self.i_mu = 9
        self.i_sigma = 0.6
        self.income = np.random.lognormal(self.i_mu, self.i_sigma, 1)

        self.repair_cost = 0
        self.resell_cost = 0
        self.landfill_cost = 0
        self.hoard_cost = 0
        self.recycle_cost = 0

        self.max_time_hoard = max_time_hoard
        self.num_cumulative_purchase_new = 0
        self.num_cumulative_purchase_used = 0

        # column sum up to 1
        self.weight_att_eol = {"repair": 0.005, "resell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425}
        self.weight_sn_eol = {"repair": 0.005, "resell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425}
        self.weight_pbc_eol = {"repair": 0.005, "resell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425}

        # column sum up to 1
        self.weight_att_purchase = {'used': 0.5, 'new': 0.5}
        self.weight_sn_purchase = {'used': 0.25, 'new': 0.25}
        self.weight_pbc_purchase = {'used': 0.25, 'new': 0.25}

        self.eol_pathway_choices = ["repair", "resell", "recycle", "landfill", "hoard"]
        self.purchase_choices = ["used", "new"]

        self.pbc_costs_purchase = {'used': 3000, 'new': 4000}
        self.pbc_costs_eol = {"repair": 0.005, "resell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425} 

        self.behavior_intention = {}
        
        # Get ID for recyclers and manufacturer
        self.recycling_facility_id = model.num_consumers + random.randrange(model.num_recyclers)
        self.refurbisher_id = model.num_consumers + model.num_prod_n_recyc + \
            random.randrange(model.num_sechdstores)

    def update_income(self, growth_rate=0.05):
        """
        Update consumer incomes using a log-normal distribution and Matthew effect.

        Args:
            mu (float): Mean of the log-normal distribution.
            sigma (float): Standard deviation of the log-normal distribution.
            growth_rate (float): Overall income growth rate.
        """
        increments = np.random.lognormal(self.i_mu, self.i_sigma, 1)  # Income increments
        # Matthew effect: allocate increments more likely to wealthier individuals
        probabilities = self.income / np.sum(self.model.all_comsumer_income)  # Wealthier individuals get larger share
        self.income += growth_rate * increments * probabilities

    def tpb_attitude(self, decision, att_level_env, weight_att):
        """
        Calculate pro-environmental attitude component of EoL TPB rule. Options
        considered pro environmental get a higher score than other options.
        Parameters:
            decision (str): The type of decision being made (e.g., "eol_pathway" or "purchase_choice").
            att_level_reuse (float): The attitude level towards reuse/pro-environmental options.
            weight_att (dict): A dictionary of weights for each option in the attitude calculation.
        Returns:
            att_level_ratios (dict): A dictionary containing the calculated attitude levels for each option,
                                     weighted by the corresponding attitude weight.
        """
        att_level_ratios = {}
        if decision == "eol_pathway":
            for i, pathway in enumerate(self.eol_pathway_choices):
                if pathway in ["repair", "resell", "recycle"]:
                    att_level_ratios[pathway] = att_level_env * weight_att[pathway]
                else:
                    att_level_ratios[pathway] = (1 - att_level_env) * weight_att[pathway]
        
        elif decision == "purchase_choice":
            for i, pathway in enumerate(self.purchase_choices):
                if self.purchase_choices[i] == 'used':
                    purchase_ratio = self.pbc_costs_purchase['used'] / (self.income)
                    att_level_ratios[pathway] = max(att_level_env - purchase_ratio, 0) * weight_att[pathway]
                else:
                    purchase_ratio = self.pbc_costs_purchase['new'] / (self.income)
                    att_level_ratios[pathway] = max(1 - att_level_env - purchase_ratio, 0) * weight_att[pathway]
        return att_level_ratios

    def tpb_subjective_norm(self, decision, weight_sn):
        """
        Parameters:
            decision (str): The type of decision being made (e.g., "repair, resell, recycle" or "used, new").
            weight_sn (dict): A dictionary of weights for each choice in the subjective norm calculation.
        Returns:
            proportion_sn (dict): A dictionary containing the proportion of neighbors making each choice, 
                weighted by the corresponding subjective norm weight.
        Calculate subjective norm (peer pressure) component of TPB rule
        """
        neighbor_agents = self.model.grid.get_neighbors(
            self.pos, include_center=False, radius=1)
        neighbor_agents = [agent for agent in neighbor_agents if isinstance(agent, Consumer)]
        proportion_sn = {}
        if decision == "eol_pathway":
            list_choices = self.eol_pathway_choices
        else:
            list_choices = self.purchase_choices
            
        for choice in list_choices:
            # for agent in neighbor_agents:
            #     print(f'I am {self.unique_id}, Neighbour: {agent.pathway_choice}')
            proportion_choice = sum(1 for agent in neighbor_agents
                                    if agent.pathway_choice == choice) / len(neighbor_agents)
            # print(choice, proportion_choice)
            proportion_sn[choice] = proportion_choice * weight_sn[choice]
        return proportion_sn

    def tpb_perceived_behavioral_control(self, decision, pbc_costs, weight_pbc):
        """
        Parameters:
            decision (str): The type of decision being made (e.g., "eol_pathway" or "purchase_choice").
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
        elif decision == "eol_pathway":
            pbc_eol = {}
            for key in pbc_costs.keys():
                pbc_eol[key] = pbc_costs[key] / max_cost
            max_eol = max(abs(i) for i in pbc_eol.values())
            return {key: -1 * value / max_eol * weight_pbc[key] for key, value in pbc_eol.items()}

    def tpb_decision(self, decision, weight_att, weight_sn, weight_pbc,
                     pbc_costs, att_level_reuse):
        """
        Select the decision with highest behavioral intention following the
        Theory of `Planned Bahevior` (TPB). `Behavioral intention` is a function
        of the subjective norm, the perceived behavioral control and attitude.
        Parameters:
            decision (str): The type of decision being made (e.g., "eol_pathway" or "purchase_choice").
            pbc_costs (dict): A dictionary containing the costs associated with each option.
        """
        # Subjective norm (peer pressure)
        sn_values = self.tpb_subjective_norm(decision, weight_sn)
        # Perceived behavioral control
        pbc_values = self.tpb_perceived_behavioral_control(decision, pbc_costs, weight_pbc)
        # Pro-environmental attitude
        att_values = self.tpb_attitude(
            decision=decision,
            att_level_env=att_level_reuse,
            weight_att=weight_att)
        
        if decision == "eol_pathway":
            list_choices = self.eol_pathway_choices
        else:
            list_choices = self.purchase_choices
            
        self.behavior_intention = {}
        for choice in list_choices:
            self.behavior_intention[choice] = pbc_values[choice] + sn_values[choice] + att_values[choice]
            # print(choice, pbc_values[choice], sn_values[choice], att_values[choice])
        
        self.pathway_choice = max(self.behavior_intention, key=self.behavior_intention.get)
        
        # If hoarding time >= max_time_hoard, then choose to reresell, recycle or landfill.
        if self.pathway_choice == "hoard" and self.smartphone.time_held >= self.max_time_hoard: # exceed max_time_hoard
            valid_choices = ["resell", "recycle", "landfill"]
            return max(valid_choices, key=lambda k: self.behavior_intention[k])
        
        # #### Print function
        # if decision == "eol_pathway":
        #     print(f'Consumer {self.unique_id} decides to {self.pathway_choice} the smartphone.')
        # else:
        #     print(f'Consumer {self.unique_id} decides to purchase a {self.pathway_choice} smartphone.')

    def use_smartphone(self):
        """
        Simulate the consumer using the smartphone and updating its state.
        """
        if self.smartphone:
            self.smartphone.update_time_held()

    def get_eol_cost_from_smartphone(self):
        """
        Calculate and set the costs for different end-of-life options.
        for perceived_behavioral_control in TPB model.
        """
        self.repair_cost = self.smartphone.calculate_repair_cost() # repair cost need to be paid by consumer
        self.resell_cost = -self.smartphone.calculate_resell_price() # reresell cost is paid by second-hand store
        self.recycle_cost = -self.smartphone.calculate_recycle_price() # recycle cost is paid by second-hand store
        self.landfill_cost = 10
        self.hoard_cost = 0

        self.pbc_costs_eol = {
            "repair": self.repair_cost / self.income, 
            "resell": self.resell_cost / self.income, 
            "recycle": self.recycle_cost / self.income, 
            "landfill": self.landfill_cost, 
            "hoard": self.hoard_cost
            }

    def purchase_smartphone(self):
        """
        Simulate the purchase of a smartphone from the market.
        Parameters:
            market (str): Indicates whether to purchase from the "new" or "used" market.
        """
        #======================== Purchase New Phone ========================#
        if self.pathway_choice == "new":
            manufacutrers = [agent for agent in self.model.agents 
                             if isinstance(agent, Manufacturer)]
            seller = random.choice(manufacutrers)
            self.smartphone = seller.trade_with_consumer(consumer_id=self.unique_id)
            print(f'Consumer {self.unique_id} buy a new phone from Producer {seller.unique_id}')
            #             consumer_id=self.unique_id)
            # for seller in manufacutrers:
            #     if abs(self.income - seller.product_price) < 200:
            #         self.smartphone = seller.trade_with_consumer(
            #             consumer_id=self.unique_id)
            #         print(f'Consumer {self.unique_id} buy a new phone from Producer {seller.unique_id}')
            #         break

            # if self.smartphone is None:
            #     min_price = 100000
            #     min_price_seller = None
            #     for seller in manufacutrers:
            #         if seller.product_price < min_price:
            #             min_price = seller.product_price
            #             min_price_seller = seller
            #     self.smartphone = min_price_seller.trade_with_consumer(
            #             consumer_id=self.unique_id)
                # print(f'Consumer {self.unique_id} buy a new phone from Producer {min_price_seller.unique_id}')

        #======================== Purchase Used Phone ========================#
        elif self.pathway_choice == "used":
            sechdstores = [agent for agent in self.model.agents 
                             if isinstance(agent, SecondHandStore)]
            seller = random.choice(sechdstores)
            self.smartphone = seller.trade_with_consumer_resell(consumer_id=self.unique_id)
            print(f'Consumer {self.unique_id} buy a used phone from second-hand store {seller.unique_id}')

    def resell_smartphone(self, new_owner_id):
        """
        Simulate the reselling of the smartphone.
        """
        sechdstores = [agent for agent in self.model.agents 
                             if isinstance(agent, SecondHandStore)]
        self.smartphone.resell_product(new_owner_id)
        # self.model.second_market
        # 更改产品的所有权
        self.smartphone = None
        # print(f"Consumer {self.consumer_id} sold their smartphone.")

    def recycle_smartphone(self, new_owner_id):
        """
        Simulate the recycling of the smartphone.
        """
        recyclers = [agent for agent in self.model.agents 
                             if isinstance(agent, Recycler)]
        self.smartphone.recycle_product(new_owner_id)
        # 更改产品的所有权
        self.smartphone = None
        # print(f"Consumer {self.consumer_id} recycled their smartphone.")

    def landfill_smartphone(self):
        """
        Simulate the landfilling of the smartphone.
        """
        self.smartphone = None
        # print(f"Consumer {self.consumer_id} landfilled their smartphone.")

    def repair_smartphone(self):
        """
        Simulate the repairing of the smartphone.
        """
        self.smartphone.repair_product()
        # print(f"Consumer {self.consumer_id} repaired their smartphone.")

    def step(self):
        """
        Main simulation step for the consumer.
        """
        # print(f"Consumer {self.unique_id} doing.")
        if self.model.steps % 12 == 0 and self.model.steps != 0:
            self.update_income()
            # print(f"Update Consumer {self.unique_id} with income {self.income}")
        
        # Step 1: Check if the consumer needs a new smartphone and update 'self.to_purchase'
        self.to_purchase = self.smartphone is None
        # print(f"Consumer {self.unique_id} to purchase: {self.to_purchase}")
        
        if self.to_purchase:
            # Step 2.1: Decide whether to purchase new or used smartphone
            # update 'self.pathway_choice'
            self.tpb_decision(
                decision='purchase_choice',
                weight_att=self.weight_att_purchase,
                weight_sn=self.weight_sn_purchase,
                weight_pbc=self.weight_pbc_purchase,
                pbc_costs=self.pbc_costs_purchase,
                att_level_reuse=float(np.random.normal(0.6, 0.1)))
            
            self.purchase_smartphone()

            if self.pathway_choice == 'used':
                self.num_cumulative_purchase_used += 1
            else:
                self.num_cumulative_purchase_new += 1
            self.use_smartphone()
        
        else:
            # Step 2.2: Use the smartphone and check EoL decision
            # update 'self.pathway_choice'
            self.get_eol_cost_from_smartphone()
            self.pathway_choice = self.tpb_decision(
                decision='eol_pathway',
                weight_att=self.weight_att_eol,
                weight_sn=self.weight_sn_eol,
                weight_pbc=self.weight_pbc_eol,
                pbc_costs=self.pbc_costs_eol,
                att_level_reuse=float(np.random.normal(0.4, 0.1)))
            
            if self.pathway_choice == "hoard":
                # update the time held and performance
                self.use_smartphone()

            elif self.pathway_choice == "resell":
                self.resell_smartphone(new_owner_id=None)

            elif self.pathway_choice == "recycle":
                self.recycle_smartphone(new_owner_id=None)

            elif self.pathway_choice == "landfill":
                self.landfill_smartphone()
                
            elif self.pathway_choice == "repair":
                self.repair_smartphone()
