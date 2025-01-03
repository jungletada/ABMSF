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
        self.preference = random.choice([agent.unique_id for agent in self.model.agents
                             if isinstance(agent, Manufacturer)])

        self.i_mu = 9
        self.i_sigma = 0.6
        self.income = np.random.lognormal(self.i_mu, self.i_sigma, 1)

        self.repair_cost = 0
        self.resell_cost = 0
        self.landfill_cost = 10
        self.hoard_cost = 10
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
        Update consumer's income annually with growth rate and Matthew effect.

        Args:
            growth_rate (float): Annual income growth rate, defaults to 0.05 (5%)
        """
        if self.model.steps % 12 == 0 and self.model.steps != 0:
            increments = np.random.lognormal(self.i_mu, self.i_sigma, 1)  # Income increments
            # Matthew effect: allocate increments more likely to wealthier individuals
            probabilities = self.income / np.sum(self.model.all_comsumer_income)  # Wealthier individuals get larger share
            self.income += growth_rate * increments * probabilities
            self.income = math.ceil(self.income)

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
            return {key: -1 * value / max_cost * weight_pbc[key] 
                        for key, value in pbc_costs.items()}
        
        # PBC for EoL pathways
        pbc_eol = {}
        for key in pbc_costs.keys():
            pbc_eol[key] = pbc_costs[key] / max_cost
        max_eol = max(abs(i) for i in pbc_eol.values())
        
        return {key: -1 * value / max_eol * weight_pbc[key] 
                    for key, value in pbc_eol.items()}

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
            self.behavior_intention[choice] = \
                pbc_values[choice] + sn_values[choice] + att_values[choice]
            # print(choice, pbc_values[choice], sn_values[choice], att_values[choice])
        
        self.pathway_choice = max(self.behavior_intention,
                                key=self.behavior_intention.get)
        
        # If hoarding time >= max_time_hoard, then choose to reresell, recycle or landfill.
        if self.pathway_choice == "hoard" and \
            self.smartphone.time_held >= self.max_time_hoard: # exceed max_time_hoard
            valid_choices = ["resell", "recycle", "landfill"]
            return max(valid_choices, key=lambda k: self.behavior_intention[k])

        # Print function
        if decision == "eol_pathway":
            print(f'Consumer {self.unique_id} decides to {self.pathway_choice} the smartphone.')
        else:
            print(f'Consumer {self.unique_id} decides to purchase a {self.pathway_choice} smartphone.')

    def use_smartphone(self):
        """
        Update smartphone usage time and state for the current consumer.
        """
        if self.smartphone:
            self.smartphone.update_time_held()

    def get_eol_cost_from_smartphone(self):
        """
        Calculate end-of-life costs for different disposal options based on smartphone condition.
        Updates the PBC costs used in TPB decision making.
        """
        # repair cost need to be paid by consumer
        self.repair_cost = self.smartphone.calculate_repair_cost()
        # reresell cost is paid by second-hand store
        self.resell_cost = -self.smartphone.calculate_resell_price_sechnd()
        # recycle cost is paid by second-hand store
        self.recycle_cost = -self.smartphone.calculate_recycle_price()
        self.pbc_costs_eol = {
            "repair": self.repair_cost / self.income, 
            "resell": self.resell_cost / self.income, 
            "recycle": self.recycle_cost / self.income, 
            "landfill": self.landfill_cost, 
            "hoard": self.hoard_cost}

    def purchase_smartphone_from_manufacturer(self):
        """
        Execute smartphone purchase based on pathway choice (new/used).
        Handles transactions with manufacturers or second-hand stores.
        """
        #======================== Purchase New Phone ========================#
        if self.pathway_choice == "new":
            manufacutrers = [agent for agent in self.model.agents
                             if isinstance(agent, Manufacturer)]
            seller = random.choice(manufacutrers)
            # Purchase smartphone from the manufacturer
            self.smartphone = seller.trade_with_consumer(consumer_id=self.unique_id)
            print(f'Consumer {self.unique_id} buy a new phone from Producer {seller.unique_id}')

        #======================== Purchase Used Phone ========================#
        elif self.pathway_choice == "used":
            sechdstores = [agent for agent in self.model.agents 
                             if isinstance(agent, SecondHandStore)]
            seller = random.choice(sechdstores)
            self.smartphone = seller.trade_with_consumer_resell(consumer_id=self.unique_id)
            # print(f'Consumer {self.unique_id} buy a used phone from second-hand store {seller.unique_id}')
        else:
            print(f"pathway_choice={self.pathway_choice} is not available.")
            raise NotImplementedError

    def resell_smartphone_to_second_store(self):
        """
        Sell current smartphone to a selected second-hand store.
        """
        sechdstores = [agent for agent in self.model.agents 
                            if isinstance(agent, SecondHandStore)]
        seller = random.choice(sechdstores)
        seller.buy_from_consumer(self.smartphone)
        self.smartphone = None
        # print(f"Consumer {self.consumer_id} sold their smartphone.")

    def recycle_smartphone(self):
        """
        Send current smartphone to a selected recycler or manufacturer.
        """
        recyclers = [agent for agent in self.model.agents
                             if isinstance(agent, Recycler)]
        manufacutrers = [agent for agent in self.model.agents
                             if isinstance(agent, Manufacturer)]
        all_recyclers = manufacutrers + recyclers
        trader = random.choice(all_recyclers)
        trader.recycle_from_customer(self.smartphone, self.unique_id)
        self.smartphone = None
        # print(f"Consumer {self.consumer_id} recycled their smartphone.")

    def purchase_with_manufacturer(self):
        # Get all manufacturers
        manufacturers = [agent for agent in self.model.schedule.agents 
                         if isinstance(agent, Manufacturer)]
        # Evaluate smartphones based on price and features
        utilities = {}
        income2price = {}
        
        for manufacturer in manufacturers:
            income2price[manufacturer.unique_id] = self.income / manufacturer.product_price
        
        # Calculate min and max
        values = list(income2price.values())
        min_value = min(values)
        max_value = min(max(values), 1)
        # Apply min-max normalization formula
        n_income2price = {
            key: (value - min_value) / (max_value - min_value) 
                for key, value in income2price.items()
        }

        noise = np.random.normal(0, 0.02)
        for manufacturer in manufacturers:
            utilities[manufacturer.unique_id] \
                    = 0.5 * n_income2price[manufacturer.unique_id] \
                    + 0.3 * self.preference \
                    + 0.2 * manufacturer.features2price \
                    + noise
            
        best_utility_id = max(utilities, key=utilities.get)
        best_utility = utilities[best_utility_id]

        trader = self.model.schedule._agents[best_utility_id]
        self.smartphone = trader.trade_with_consumer(self.unique_id)
        print(f"Consumer {self.unique_id} purchased smartphone: {self.smartphone}")
    
    def purchase_with_second_store(self):
        # Get all manufacturers
        sechdstores = [agent for agent in self.model.schedule.agents
                         if isinstance(agent, SecondHandStore)]
        view_size = int(0.4 * self.model.num_sechdstores)
        sechdstores = random.choices(sechdstores, k=view_size)

        # Evaluate smartphones based on price and features
        utilities = {}
        income2price = {}
        for second_store in sechdstores:
            if len(second_store.inventory) != 0:
                for smartphone in second_store.inventory:
                    used_price = smartphone.calculate_sechnd_market_price
                    income2price[smartphone.product_id] = self.income / used_price

        # Calculate min and max
        values = list(income2price.values())
        min_value = min(values)
        max_value = min(max(values), 1)
        # Apply min-max normalization formula
        n_income2price = {
            key: (value - min_value) / (max_value - min_value) 
                for key, value in income2price.items()}

        best_utility_id = None
        best_product_id = None
        best_utility = -float(math.inf)
        noise = np.random.normal(0, 0.02)
        for second_store in sechdstores:
            if len(second_store.inventory) != 0:
                for smartphone in second_store.inventory:
                    producer = self.model.schedule._agents[smartphone.producer_id]
                    utility = 0.5 * n_income2price[smartphone.product_id] \
                            + 0.3 * self.preference \
                            + 0.2 * producer.features2price \
                            + noise
                    if utility > best_utility:
                        best_utility_id = second_store.unique_id
                        best_utility = utility
                        best_product_id = smartphone.product_id
        # best_utility_id = max(utilities, key=utilities.get)
        # best_utility = utilities[best_utility_id]
        trader = self.model.schedule._agents[best_utility_id]
        self.smartphone = trader.trade_with_consumer_resell(
            consumer_id=self.unique_id, product_id=best_product_id)
        print(f"Consumer {self.unique_id} purchased smartphone: {self.smartphone}")
    
    def calculate_recycling_intention(self):
        """
        Calculate recycling intention based on the extended TPB model formula.
        """
        # Weights for the extended TPB model
        self.w_att_recycle = 0.45
        self.w_sn_recycle = 0.35
        self.w_pbc_recycle = 0.35
        self.w_mn_recycle = 0.20
        self.w_pc_recycle = 0.15
        self.w_md_recycle = 0.20
        # Compute intention score
        recycling_intention = (
            self.w_att_recycle * self.att_recycle +
            self.w_sn_recycle * self.sn_recycle +
            self.w_pbc_recycle * self.pbc_recycle +
            self.w_mn_recycle * self.moral__recycle -
            self.w_pc_recycle * self.privacy__recycle -
            self.w_md_recycle * self.moderate_recycle
        )
        return recycling_intention

    def decide_recycling(self, threshold=0.5):
        """
        Decide whether to participate in recycling based on the intention score.
        """
        intention_score = self.calculate_recycling_intention()
        return intention_score >= threshold

    def landfill_smartphone(self):
        """
        Dispose of current smartphone in landfill.
        """
        if self.smartphone is not None:
            self.smartphone.remove()
        self.smartphone = None
        # print(f"Consumer {self.consumer_id} landfilled their smartphone.")

    def repair_smartphone(self):
        """
        Repair current smartphone to improve its condition.
        """
        self.smartphone.repair_product()
        # print(f"Consumer {self.consumer_id} repaired their smartphone.")

    def step(self):
        """
        Main simulation step for the consumer.
        """
        # print(f"Consumer {self.unique_id} doing.")
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
            
            self.purchase_smartphone_from_manufacturer()

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
                self.resell_smartphone_to_second_store()

            elif self.pathway_choice == "recycle":
                self.recycle_smartphone()

            elif self.pathway_choice == "landfill":
                self.landfill_smartphone()
                
            elif self.pathway_choice == "repair":
                self.repair_smartphone()
