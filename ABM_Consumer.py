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
            preference_id,
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
        self.pathway_action = 'new'
        self.preference_id = preference_id
        # income part
        self.i_mu = 8
        self.i_sigma = 0.44
        self.income = int(np.random.lognormal(
            self.i_mu, self.i_sigma, 1))

        self.repair_cost = 0
        self.resell_cost = 0
        self.landfill_cost = 10
        self.hoard_cost = 10
        self.recycle_cost = 0

        self.max_time_hoard = max_time_hoard
        self.num_cumulative_purchase_new = 0
        self.num_cumulative_purchase_used = 0

        # column sum up to 1
        self.eol_choices = ["repair", "resell", "recycle", "landfill", "hoard"]
        self.weight_att_eol = {self.eol_choices[0]: 0.005,
                               self.eol_choices[1]: 0.01,
                               self.eol_choices[2]: 0.1,
                               self.eol_choices[3]: 0.4425,
                               self.eol_choices[4]: 0.4425}
        self.weight_sn_eol = {self.eol_choices[0]: 0.005,
                               self.eol_choices[1]: 0.01,
                               self.eol_choices[2]: 0.1,
                               self.eol_choices[3]: 0.4425,
                               self.eol_choices[4]: 0.4425}
        self.weight_pbc_eol = {self.eol_choices[0]: 0.005,
                               self.eol_choices[1]: 0.01,
                               self.eol_choices[2]: 0.1,
                               self.eol_choices[3]: 0.4425,
                               self.eol_choices[4]: 0.4425}
        self.pbc_costs_eol = {self.eol_choices[0]: 0.005,
                               self.eol_choices[1]: 0.01,
                               self.eol_choices[2]: 0.1,
                               self.eol_choices[3]: 0.4425,
                               self.eol_choices[4]: 0.4425}

        # column sum up to 1
        self.purchase_choices = ["used", "new"]
        self.weight_att_purchase = {'used': 0.5, 'new': 0.5}
        self.weight_sn_purchase = {'used': 0.25, 'new': 0.25}
        self.weight_pbc_purchase = {'used': 0.25, 'new': 0.25}
        self.pbc_costs_purchase = {'used': 0, 'new': 0}

        self.behavior_intention = {}
        # Recycle intention
        self.trade_in_id = None
        self.w_att_rc = 0.45
        self.w_sn_rc = 0.35
        self.w_pbc_rc = 0.35
        self.w_mn_rc = 0.20
        self.w_pc_rc = 0.15
        self.w_md_rc = 0.20

        self.w_att_pc_rc = 0.25
        self.w_sn_pc_rc = 0.25
        self.w_pbc_pc_rc = 0.25
        self.w_mn_pc_rc = 0.25

        self.recycle_choices = ['manufacturer', 'recycler']
        self.att_recycle = {self.recycle_choices[0]: 0.5, self.recycle_choices[1]: 0.5}
        self.sn_recycle =  {self.recycle_choices[0]: 0.5, self.recycle_choices[1]: 0.5}
        self.pbc_recycle = {self.recycle_choices[0]: 0.5, self.recycle_choices[1]: 0.5}
        self.mn_recycle =  {self.recycle_choices[0]: 0.5, self.recycle_choices[1]: 0.5}
        self.pc_recycle =  {self.recycle_choices[0]: 0.5, self.recycle_choices[1]: 0.5}
        self.md_recycle =  {self.recycle_choices[0]: 0.5, self.recycle_choices[1]: 0.5}
        self.recycling_intention = {}
        self.recycle_action = None
        # Get ID for recyclers and manufacturer
        self.recycling_facility_id = model.num_consumers + random.randrange(model.num_recyclers)
        self.refurbisher_id = model.num_consumers + model.num_prod_n_recyc + \
            random.randrange(model.num_sechdstores)
        
    def update_cost(self):
        sechdstores = self.model.agents_by_type[SecondHandStore]
        view_size = int(0.4 * self.model.num_sechdstores)
        sechdstores = random.choices(sechdstores, k=view_size)
        num_stocks = 0
        total_used_prices = 0
        for second_store in sechdstores:
            if len(second_store.inventory) != 0:
                for smartphone in second_store.inventory:
                    used_price = smartphone.calculate_sechnd_market_price()
                    num_stocks += 1
                    total_used_prices += used_price
        if num_stocks != 0:
            local_avg_used_product_price = int(total_used_prices / num_stocks)
        else:
            local_avg_used_product_price = None
        self.pbc_costs_purchase = {
            'used': local_avg_used_product_price,
            'new': self.model.avg_new_product_price}
        print(self.pbc_costs_purchase)
        
    def update_income(self, growth_rate=0.1):
        """
        Update consumer's income annually with growth rate and Matthew effect.

        Args:
            growth_rate (float): Annual income growth rate, defaults to 0.05 (5%)
        """
        
        increments = np.random.lognormal(self.i_mu, self.i_sigma, 1)  # Income increments
        # Matthew effect: allocate increments more likely to wealthier individuals
        probabilities = self.income / sum(self.model.all_comsumer_income)  # Wealthier individuals get larger share
        self.income += growth_rate * increments
        self.income = int(self.income)

    def tpb_attitude(self, decision, att_level_env, weight_att):
        """
        Calculates the pro-environmental attitude component of the Theory of Planned Behavior (TPB) rule. It assigns a higher score to options that are considered pro-environmental.
        
        Parameters:
            decision (str): Specifies the type of decision being made, such as "eol_pathway" for end-of-life pathway choices or "purchase_choice" for purchasing decisions.
            att_level_env (float): Represents the individual's attitude level towards pro-environmental options, such as reuse or recycling.
            weight_att (dict): A dictionary containing weights for each option in the attitude calculation, used to adjust the score based on the option's environmental impact.
        
        Returns:
            att_level_ratios (dict): A dictionary that contains the calculated attitude levels for each option, adjusted by the corresponding weight from the weight_att dictionary. This output represents the individual's attitude towards each option, with pro-environmental options receiving a higher score.
        """
        att_level_ratios = {}
        if decision == "eol_pathway":
            for i, pathway in enumerate(self.eol_choices):
                if pathway in ["repair", "resell", "recycle"]:
                    att_level_ratios[pathway] = att_level_env * weight_att[pathway]
                else:
                    att_level_ratios[pathway] = (1 - att_level_env) * weight_att[pathway]
        
        elif decision == "purchase_choice":
            for i, pathway in enumerate(self.purchase_choices):
                if self.purchase_choices[i] == 'used':
                    if self.pbc_costs_purchase['used'] is None:
                        att_level_ratios[pathway] = float('-inf') # Negative infinity means no stocks for used smartphone
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
            list_choices = self.eol_choices
        else:
            list_choices = self.purchase_choices
            
        for choice in list_choices:
          
            proportion_choice = sum(1 for agent in neighbor_agents
                                    if agent.pathway_action == choice) / len(neighbor_agents)
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
        Theory of `Planned Behavior` (TPB). `Behavioral intention` is a function
        of the subjective norm, the perceived behavioral control and attitude.
        Parameters:
            decision (str): The type of decision being made (e.g., "eol_pathway" or "purchase_choice").
            pbc_costs (dict): A dictionary containing the costs associated with each option.
        """
        # Perceived behavioral control
        pbc_values = self.tpb_perceived_behavioral_control(decision, pbc_costs, weight_pbc)
        # Subjective norm (peer pressure)
        sn_values = self.tpb_subjective_norm(decision, weight_sn)
        # Pro-environmental attitude
        att_values = self.tpb_attitude(
            decision=decision,
            att_level_env=att_level_reuse,
            weight_att=weight_att)
        
        if decision == "eol_pathway":
            list_choices = self.eol_choices
        else:
            list_choices = self.purchase_choices
            
        self.behavior_intention = {}
        for choice in list_choices:
            self.behavior_intention[choice] = \
                pbc_values[choice] + sn_values[choice] + att_values[choice]
            # if decision != "eol_pathway":
            #     print(f'Consumer {self.unique_id}, {choice}, {self.behavior_intention[choice]}, ' 
            #       f'{pbc_values[choice]}, {sn_values[choice]}, {att_values[choice]}')
        
        self.pathway_action = max(self.behavior_intention,
                                key=self.behavior_intention.get)
        
        # If hoarding time >= max_time_hoard, then choose to reresell, recycle or landfill.
        if self.pathway_action == "hoard" and \
            self.smartphone.time_held >= self.max_time_hoard: # exceed max_time_hoard
            valid_choices = ["resell", "recycle", "landfill"]
            return max(valid_choices, key=lambda k: self.behavior_intention[k])

        # if decision == "eol_pathway":
        #     print(f'Consumer {self.unique_id} decides to {self.pathway_action} the smartphone.')
        # else:
        #     print(f'Consumer {self.unique_id} decides to purchase a {self.pathway_action} smartphone.')

    def get_eol_cost_of_smartphone(self):
        """
        This function calculates the end-of-life costs for different disposal options based on the condition of the smartphone.
        It also updates the Perceived Behavioral Control (PBC) costs used in the Theory of Planned Behavior (TPB) decision making.
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

    def use_smartphone(self):
        """
        Update smartphone usage time and state for the current consumer.
        """
        if self.smartphone:
            self.smartphone.update_time_held()

    def purchase_smartphone(self):
        """
        Facilitates the purchase of a smartphone based on the consumer's chosen pathway, either new or used. 
        This function orchestrates the transaction process with either a manufacturer for a new phone 
        or a second-hand store for a used phone.
        """
        if self.trade_in_id is not None:
            trader = self.model._agents[self.trade_in_id]
            self.smartphone = trader.trade_with_consumer(self.unique_id)
            self.trade_in_id = None
        else:
            #======================== Purchase New Phone ========================#
            if self.pathway_action == "new":
                self.purchase_with_manufacturer()
            #======================== Purchase Used Phone ========================#
            elif self.pathway_action == "used":
                self.purchase_with_secondhand_store()
            else:
                print(f"pathway_choice={self.pathway_action} is not available.")
                raise NotImplementedError

    def dispose_smartphone_in_landfill(self):
        """
        Disposing the current smartphone in a landfill.
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

    def purchase_with_manufacturer(self):
        """
        Purchase smartphone from manufacturer based on utility function.
        Evaluates manufacturers based on income-to-price ratio, consumer preferences,
        and product features to select the best option.
        """
        # Get all manufacturers
        manufacturers = self.model.agents_by_type[Manufacturer]
        #[agent for agent in self.model.agents if isinstance(agent, Manufacturer)]
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
            utilities[manufacturer.unique_id] = \
                  0.5 * n_income2price[manufacturer.unique_id] \
                + 0.3 * int(self.preference_id == manufacturer.unique_id)\
                + 0.2 * manufacturer.features2price \
                + noise
            
        best_utility_id = max(utilities, key=utilities.get)
        best_utility = utilities[best_utility_id]
        
        for trader in manufacturers:
            if trader.unique_id == best_utility_id:
                self.smartphone = trader.trade_with_consumer(self.unique_id)
        # print(f"Consumer {self.unique_id} purchased smartphone: {self.smartphone}")

    def purchase_with_secondhand_store(self):
        """
        Purchase used smartphone from second-hand store based on utility function.
        
        Evaluates available used smartphones from a random subset of stores based on 
        income-to-price ratio, preferences, and product features.
        """
        # Get all manufacturers
        sechdstores = self.model.agents_by_type[SecondHandStore]
        view_size = int(0.4 * self.model.num_sechdstores)
        sechdstores = random.choices(sechdstores, k=view_size)
        # Evaluate smartphones based on price and features
        utilities = {}
        income2price = {}
        no_inventory = True
        for second_store in sechdstores:
            if len(second_store.inventory) != 0:
                no_inventory = False
                for smartphone in second_store.inventory:
                    used_price = smartphone.calculate_sechnd_market_price()
                    income2price[smartphone.producer_id] = self.income / used_price
        if no_inventory:
            return
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
        target_smartphone = None
        for second_store in sechdstores:
            if len(second_store.inventory) != 0:
                for smartphone in second_store.inventory:
                    utility = 0.5 * n_income2price[smartphone.producer_id] \
                            + 0.3 * int(self.preference_id == smartphone.producer_id)\
                            + 0.2 * smartphone.sec_features2price \
                            + noise
                    if utility > best_utility:
                        target_smartphone = smartphone
                        best_utility = utility
                        best_utility_id = second_store.unique_id

        for trader in sechdstores:
            if trader.unique_id == best_utility_id:
                trader.trade_with_consumer_resell(smartphone_id=target_smartphone.unique_id)
                target_smartphone.user_id = self.unique_id
                self.smartphone = target_smartphone
        # print(f"Consumer {self.unique_id} purchased smartphone: {self.smartphone}")

    def sell_smartphone_to_second_hand_store(self):
        """
        Initiates the process of selling the consumer's current smartphone to a randomly chosen second-hand store.
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

    def calculate_recycling_intention(self):
        """
        Calculates the consumer's intention to recycle their smartphone based on 
        the extended Theory of Planned Behavior (TPB) model. It considers factors such as 
        perceived behavioral control, subjective norm, moral norm, and privacy concern 
        to determine the likelihood of recycling.
        """
        tiv = self.smartphone.calculate_trade_in_value()
        rec = self.smartphone.calculate_recycle_price()
        self.pbc_recycle[self.recycle_choices[0]] = tiv / (tiv + rec)
        self.pbc_recycle[self.recycle_choices[1]] = rec / (tiv + rec)

        neighbor_agents = self.model.grid.get_neighbors(
            self.pos, include_center=False, radius=1)
        neighbor_agents = [agent for agent in neighbor_agents if isinstance(agent, Consumer)]
        
        for c in self.recycle_choices:
            self.sn_recycle[c] = sum(1 for agent in neighbor_agents
                                if agent.recycle_action == c) / len(neighbor_agents)

        for c in self.recycle_choices:
            # Combine privacy concern with attention, subjective norm, 
            # perceived behavioral control, and moral norm.
            self.md_recycle[c] = \
                self.w_att_pc_rc * self.att_recycle[c] * self.pc_recycle[c] + \
                self.w_sn_pc_rc * self.sn_recycle[c] * self.pc_recycle[c] + \
                self.w_pbc_pc_rc * self.pbc_recycle[c] * self.pc_recycle[c] + \
                self.w_mn_pc_rc * self.mn_recycle[c] * self.pc_recycle[c]
            
            # Calculate recycling intention based on the Theory of Planned Behavior, including moral norm and privacy concern. 
            self.recycling_intention[c] = \
                self.w_att_rc * self.att_recycle[c] + \
                self.w_sn_rc * self.sn_recycle[c] + \
                self.w_pbc_rc * self.pbc_recycle[c] + \
                self.w_mn_rc * self.mn_recycle[c] - \
                self.w_pc_rc * self.pc_recycle[c] - \
                self.w_md_rc * self.md_recycle[c]

    def recycling_smartphone_tpb(self):
        """
        Determines the consumer's decision to recycle their smartphone based on the 
        Theory of Planned Behavior (TPB) model. It calculates the recycling intention score 
        and selects the recycling action with the highest score. 
        Depending on the chosen action, the smartphone is either recycled with a manufacturer 
        or a randomly selected recycler.
        """
        self.calculate_recycling_intention()
        self.recycle_action = max(self.recycling_intention,
                                key=self.recycling_intention.get)
        if self.recycle_action == 'manufacturer':
            processor_id = self.smartphone.producer_id
            processor = self.model._agents[processor_id]
            self.trade_in_id = processor_id
            processor.recycle_from_customer(self.smartphone, self.unique_id)
        else:
            recyclers = [agent for agent in self.model.agents
                             if isinstance(agent, Recycler)]
            processor = random.choice(recyclers)
            processor.recycle_from_customer(self.smartphone, self.unique_id)
            self.smartphone = None
    
    def step(self):
        """
        Main simulation step for the consumer.
        """
        self.update_cost()
        if self.model.steps % 12 == 0:
            self.update_income() # update the income according to Matthew Effect
        
        # Step 1: Check if the consumer needs a new smartphone and update 'self.to_purchase'
        self.to_purchase = self.smartphone is None

        # if self.to_purchase:
        #     # Step 2.1: Decide whether to purchase new or used smartphone
        #     # update 'self.pathway_choice'
        #     self.tpb_decision(
        #         decision='purchase_choice',
        #         weight_att=self.weight_att_purchase,
        #         weight_sn=self.weight_sn_purchase,
        #         weight_pbc=self.weight_pbc_purchase,
        #         pbc_costs=self.pbc_costs_purchase,
        #         att_level_reuse=float(np.random.normal(0.6, 0.1)))
            
        #     self.purchase_smartphone()

        #     if self.pathway_action == 'used':
        #         self.num_cumulative_purchase_used += 1
        #     else:
        #         self.num_cumulative_purchase_new += 1
            
        #     self.use_smartphone()

        # else:
        #     # Step 2.2: Use the smartphone and check EoL decision
        #     # update 'self.pathway_choice'
        #     self.use_smartphone()
        #     self.get_eol_cost_of_smartphone()

        #     self.pathway_action = self.tpb_decision(
        #         decision='eol_pathway',
        #         weight_att=self.weight_att_eol,
        #         weight_sn=self.weight_sn_eol,
        #         weight_pbc=self.weight_pbc_eol,
        #         pbc_costs=self.pbc_costs_eol,
        #         att_level_reuse=float(np.random.normal(0.4, 0.1)))
            
        #     if self.pathway_action == "hoard":
        #         # update the time held and performance
        #         pass
        #     elif self.pathway_action == "resell":
        #         self.resell_smartphone_to_second_store()

        #     elif self.pathway_action == "recycle":
        #         self.recycle_smartphone()
                
        #     elif self.pathway_action == "landfill":
        #         self.landfill_smartphone()
                                
        #     elif self.pathway_action == "repair":
        #         self.repair_smartphone()
