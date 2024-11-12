import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from mesa import Model
# from mesa import AgentSet
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector


from ABM_Consumer import Consumer
from ABM_Recycler import Recycler
from ABM_SecHandStore import SecondHandStore
from ABM_Manufacturer import Manufacturer


class Smartphone_MODEL(Model):
    """

    Attributes:
        seed (to fix random numbers), (default=None). Modeler's choice.
        calibration_n_sensitivity1 (use to vary list parameters or
            dictionnaries), (default=1). Modeler's choice.
        calibration_n_sensitivity_2 (use to vary list parameters or
            dictionnaries), (default=1). Modeler's choice.
        num_consumers, (default=1000). Simplifying assumption
        consumers_node_degree, (default=5). From Small-World literature.
        consumers_network_type=("small-world", "complete graph", "random"
            "cycle graph", "scale-free graph"), (default="small-world").
            From Small-World literature (e.g., Byrka et al. 2016).
        num_recyclers, (default=16). 16 From SEIA, 2019.
        num_producers, (default=60). Simplifying assumption.
        prod_n_recyc_node_degree, (default=5). From Small-World literature.
        prod_n_recyc_network_type, (default="small-world"). From Small-World
            and industrial symbiosis literature (e.g., Doménech & Davies,
            2011).
        num_refurbishers, (default=15). From Wood Mackenzie, 2019.
        consumers_distribution (allocation of different types of consumers),
            (default={"residential": 1, "commercial": 0., "utility": 0.}).
            (Other possible values based on EIA, 2019 and SBE council, 2019:
            residential=0.75, commercial=0.2 and utility=0.05).
        init_eol_rate (dictionary with initial end-of-life (EOL) ratios),
            (default={"repair": 0.005, "sell": 0.02, "recycle": 0.1,
            "landfill": 0.4375, "hoard": 0.4375}). From Monteiro Lunardi
            et al 2018 and European Commission (2015).
        init_purchase_choice (dictionary with initial purchase ratios),
            (default={"new": 0.98, "used": 0.02, "certified": 0}). From
            European Commission (2015).
        total_number_product (a list for the whole population e.g.
            (time series of product quantity)) (in unit of functional
            unit (fu), examples of fu: Wp for PV, Bit for hard drive), (
            default=[38, 38, 38, 38, 38, 38, 38, 139, 251, 378, 739, 1670,
            2935, 4146, 5432, 6525, 3609, 4207, 4905, 5719]). From IRENA-IEA
            2016 (assuming that 87% of the PV market is c-Si).
        product_distribution (ratios of product among consumer types), (default
            ={"residential": 1, "commercial": 0., "utility": 0.}). (Other
            possible values based on Bolinger et al. 2018: residential=0.21,
            commercial=0.18 and utility=0.61).
        product_growth (a list for a piecewise function) (ratio), (default=
            [0.166, 0.045]). From IRENA-IEA 2016
        growth_threshold, (default=10). From IRENA-IEA 2016
        failure_rate_alpha (a list for a triangular distribution), (default=
            [2.4928, 5.3759, 3.93495]). From IRENA-IEA 2016.
        hoarding_cost (a list for a triangular distribution) ($/fu), (default=
            [0, 0.001, 0.0005]). From www.cisco-eagle.com (accessed 12/2019).
        landfill_cost (a list for a triangular distribution) ($/fu), (default=
            [0.0089, 0.0074, 0.0071, 0.0069, 0.0056, 0.0043,
                     0.0067, 0.0110, 0.0085, 0.0082, 0.0079, 0.0074, 0.0069,
                     0.0068, 0.0068, 0.0052, 0.0052, 0.0051, 0.0074, 0.0062,
                     0.0049, 0.0049, 0.0047, 0.0032, 0.0049, 0.0065, 0.0064,
                     0.0062, 0.0052, 0.0048, 0.0048, 0.0044, 0.0042, 0.0039,
                     0.0039, 0.0045, 0.0055, 0.0050, 0.0049, 0.0044, 0.0044,
                     0.0039, 0.0033, 0.0030, 0.0041, 0.0050, 0.0040, 0.0040,
                     0.0038, 0.0033]). From EREF 2019.
        w_sn_eol (the weight of subjective norm in the agents' decisions as
            modeled with the theory of planned behavior), (default=0.33). From
            Geiger et al. 2019.
        w_pbc_eol (the weight of perceived behavioral control in the agents'
            decisions as modeled with the theory of planned behavior), (
            default=0.39). From Geiger et al. 2019.
        w_a_eol (the weight of attitude in the agents' decisions as modeled
            with the theory of planned behavior), (default=0.34). From
            Geiger et al. 2019.
        w_sn_reuse (same as above but for remanufactured product purchase
            decision), (default=0.497). From Singhal et al. 2019.
        w_pbc_reuse (same as above but for remanufactured product purchase
            decision), (default=0.382). From Singhal et al. 2019.
        w_a_reuse (same as above but for remanufactured product purchase
            decision), (default=0.464). From Singhal et al. 2019.
        product_lifetime (years), (default=30). From IRENA-IEA 2016.
        all_EoL_pathways (dictionary of booleans for EOL pathways), (default=
            {"repair": True, "sell": False, "recycle": True, "landfill": False,
            "hoard": True}). Modeler's choice.
        max_storage (a list for a triangular distribution) (years), (default=
            [1, 8, 4]). From Wilson et al. 2017.
        att_distrib_param_eol (a list for a bounded normal distribution), (
            default=[0.53, 0.12]). From model's calibration step (mean),
            Saphores 2012 (standard deviation).
        att_distrib_param_eol (a list for a bounded normal distribution), (
            default=[0.35, 0.2]). From model's calibration step (mean),
            Abbey et al. 2016 (standard deviation).
        original_recycling_cost (a list for a triangular distribution) ($/fu) (
            default=[0.106, 0.128, 0.117]). From EPRI 2018.
        recycling_learning_shape_factor, (default=-0.39). From Qiu & Suh 2019.
        repairability (ratio), (default=0.55). From Tsanakas et al. 2019.
        original_repairing_cost (a list for a triangular distribution) ($/fu),
            (default=[0.1, 0.45, 0.28]). From Tsanakas et al. 2019.
        repairing_learning_shape_factor, (default=-0.31). Estimated with data
            on repairing costs at different scales from JRC 2019.
        scndhand_mkt_pric_rate (a list for a triangular distribution) (ratio),
            (default=[0.4, 1, 0.7]). From unpublished study Wang et al.
        fsthand_mkt_pric ($/fu), (default=0.3). From Wood Mackenzie, 2019.
        refurbisher_margin (ratio), (default=[0.03, 0.45, 0.24]). From Duvan
            & Aykaç 2008 and www.investopedia.com (accessed 03/2020).
        purchase_choices (dictionary of booleans for purchase choice), (default
            ={"new": True, "used": True, "certified": False}).
            Modeler's choice.
        init_trust_boundaries (from Ghali et al. 2017)
        social_event_boundaries (from Ghali et al. 2017)
        social_influencability_boundaries (from Ghali et al. 2017)
        trust_threshold (from Ghali et al. 2017)
        knowledge_threshold (from Ghali et al. 2017)
        willingness_threshold (from Ghali et al. 2017)
        self_confidence_boundaries (from Ghali et al. 2017)
        product_mass_fractions (dictionary containing mass fractions of
            materials composing the product), (default={"Product": 1,
            "Aluminum": 0.08, "Glass": 0.76, "Copper": 0.01, "Insulated cable"
            : 0.012, "Silicon": 0.036, "Silver": 0.00032}). From ITRPV, 2015,
            2018.
        established_scd_mkt (dictionary containing booleans regarding the
            availability of an industrial pathway for the recovered material),
            (default={"Product": True, "Aluminum": True, "Glass": True,
            "Copper": True, "Insulated cable": True, "Silicon": False,
            "Silver": False}). Modeler's choice.
        scd_mat_prices (dictionary containing lists for triangular
            distributions of secondary materials prices) ($/fu), (default={
            "Product": [np.nan, np.nan, np.nan], "Aluminum": [0.66, 1.98,
            1.32], "Glass": [0.01, 0.06, 0.035], "Copper": [3.77, 6.75, 5.75],
            "Insulated cable": [3.22, 3.44, 3.33], "Silicon": [2.20, 3.18,
            2.69], "Silver": [453, 653, 582]}). From www.Infomine.com (2019),
            copper.org (2019), USGS (2017), Bureau of Labor Statistics (2018),
            www.recyclingproductnews.com (all websites accessed 03/2020).
        virgin_mat_prices (dictionary containing lists for triangular
            distributions of virgin materials prices) ($/fu), (default={
            "Product": [np.nan, np.nan, np.nan], "Aluminum": [1.76, 2.51,
            2.14], "Glass": [0.04, 0.07, 0.055], "Copper": [4.19, 7.50, 6.39],
            "Insulated cable": [3.22, 3.44, 3.33], "Silicon": [2.20, 3.18,
            2.69], "Silver": [453, 653, 582]}). From Butler et al. (2005),
            Newlove  (2017), USGS (2017), www.infomine.com (2019), expert
            opinions (for insulated cables) (all websites accessed 03/2020).
        material_waste_ratio (dictionary containing industrial waste ratios),
            (default={"Product": 0., "Aluminum": 0., "Glass": 0., "Copper": 0.,
            "Insulated cable": 0., "Silicon": 0.4, "Silver": 0.}). From
            Hachichi 2018.
        recovery_fractions (dictionary containing recovery fractions of
            materials composing the product when recycled), (default={
            "Product": np.nan, "Aluminum": 0.92, "Glass": 0.85, "Copper": 0.72,
            "Insulated cable": 1, "Silicon": 0,
            "Silver": 0}). From Ardente et al. 2019 and IEA-PVPS task-12 2018.
        product_average_wght (kg/fu), (default=0.1). From IRENA-IEA 2016.
        mass_to_function_reg_coeff, (default=0.03). Estimated with data from
            IRENA-IEA 2016.
        recycling_states (a list of states owning at least one recycling
            facility), (default=['Texas', 'Arizona', 'Oregon', 'Oklahoma',
            'Wisconsin', 'Ohio', 'Kentucky', 'South Carolina']. From SEIA 2019.
        transportation_cost ($/t.km), (default=0.021761). From ecoinvent 2020.
        used_product_substitution_rate (a list for a triangular distribution)
            (ratio), (default=[0.6, 1, 0.8]). From unpublished study Wang et
            al.
        imperfect_substitution (model rebound effect) (ratio), (default=0).
            Modeler's choice.
        epr_business_model (boolean) (assume that enhanced producer
            responsibility means that product are recycled at the end of life
            as well as industrial waste), (default=False). Modeler's choice.
        recycling_process (dictionary of booleans), (default={"frelp": False,
            "asu": False, "hybrid": False}). Modeler's choice.
        industrial_symbiosis (boolean), (default=False). Modeler's choice.

    """

    def __init__(self,
                 seed=None,
                 calibration_n_sensitivity=1,
                 calibration_n_sensitivity_2=1,
                 num_consumers=1000,
                 consumers_node_degree=10,
                 consumers_network_type="small-world",
                 rewiring_prob=0.1,
                 num_recyclers=15,
                 num_producers=10,
                 prod_n_recyc_node_degree=5,
                 prod_n_recyc_network_type="small-world",
                 num_sechdstores=30,
                 consumers_distribution={"residential": 1, "commercial": 0., "utility": 0.},
                 init_eol_rate={"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425},
                 init_purchase_rate={"new": 0.9995, "used": 0.0005},
                 total_number_product=[38, 38, 38, 38, 38, 38, 38, 139, 251,
                                       378, 739, 1670, 2935, 4146, 5432, 6525,
                                       3609, 4207, 4905, 5719],
                 product_distribution={"residential": 1, "commercial": 0., "utility": 0.},
                 product_growth=[0.166, 0.045],
                 growth_threshold=10,
                 failure_rate_alpha=[2.4928, 5.3759, 3.93495],
                 hoarding_cost=[0, 0.001, 0.0005],
                 landfill_cost=[
                     0.0089, 0.0074, 0.0071, 0.0069, 0.0056, 0.0043,
                     0.0067, 0.0110, 0.0085, 0.0082, 0.0079, 0.0074, 0.0069,
                     0.0068, 0.0068, 0.0052, 0.0052, 0.0051, 0.0074, 0.0062,
                     0.0049, 0.0049, 0.0047, 0.0032, 0.0049, 0.0065, 0.0064,
                     0.0062, 0.0052, 0.0048, 0.0048, 0.0044, 0.0042, 0.0039,
                     0.0039, 0.0045, 0.0055, 0.0050, 0.0049, 0.0044, 0.0044,
                     0.0039, 0.0033, 0.0030, 0.0041, 0.0050, 0.0040, 0.0040,
                     0.0038, 0.0033],
                 theory_of_planned_behavior={
                     "residential": True, "commercial": True, "utility": True},
                 w_sn_eol=0.27,
                 w_pbc_eol=0.44,
                 w_a_eol=0.39,
                 w_sn_reuse=0.497,
                 w_pbc_reuse=0.382,
                 w_a_reuse=0.464,
                 product_lifetime=30,
                 all_eol_pathways={"repair": True, "sell": True,
                                   "recycle": True, "landfill": True,
                                   "hoard": True},
                 max_storage=[1, 8, 4],
                 att_distrib_param_eol=[0.544, 0.1],
                 att_distrib_param_reuse=[0.223, 0.262],
                 original_recycling_cost=[0.106, 0.128, 0.117],
                 recycling_learning_shape_factor=-0.39,
                 repairability=0.55,
                 # some modules don't need repair
                 original_repairing_cost=[0.1, 0.45, 0.23],
                 # HERE
                 repairing_learning_shape_factor=-0.31,
                 scndhand_mkt_pric_rate=[0.4, 0.2],
                 # from https://www.ise.fraunhofer.de/content/dam/ise/de/
                 # documents/publications/studies/AgoraEnergiewende_Current_and_
                 # Future_Cost_of_PV_Feb2015_web.pdf
                 # a=6.5, b=0.078, y=a*exp(b*-t)
                 fsthand_mkt_pric=0.45,
                 #0.04
                 fsthand_mkt_pric_reg_param=[1, 0.04],
                 # HERE
                 refurbisher_margin=[0.4, 0.6, 0.5],
                 purchase_choices={"new": True, "used": True, "certified": False},
                 init_trust_boundaries=[-1, 1],
                 social_event_boundaries=[-1, 1],
                 social_influencability_boundaries=[0, 1],
                 trust_threshold=0.5,
                 knowledge_threshold=0.5,
                 willingness_threshold=0.5,
                 self_confidence_boundaries=[0, 1],
                 product_mass_fractions={"Product": 1, "Aluminum": 0.08,
                                         "Glass": 0.76, "Copper": 0.01,
                                         "Insulated cable": 0.012,
                                         "Silicon": 0.036, "Silver": 0.00032},
                 established_scd_mkt={"Product": True, "Aluminum": True,
                                      "Glass": True, "Copper": True,
                                      "Insulated cable": True,
                                      "Silicon": False, "Silver": False},
                 scd_mat_prices={"Product": [np.nan, np.nan, np.nan],
                                 "Aluminum": [0.66, 1.98, 1.32],
                                 "Glass": [0.01, 0.06, 0.035],
                                 "Copper": [3.77, 6.75, 5.75],
                                 "Insulated cable": [3.22, 3.44, 3.33],
                                 "Silicon": [2.20, 3.18, 2.69],
                                 "Silver": [453, 653, 582]},
                 virgin_mat_prices={"Product": [np.nan, np.nan, np.nan],
                                 "Aluminum": [1.76, 2.51, 2.14],
                                 "Glass": [0.04, 0.07, 0.055],
                                 "Copper": [4.19, 7.50, 6.39],
                                 "Insulated cable": [3.22, 3.44, 3.33],
                                 "Silicon": [2.20, 3.18, 2.69],
                                 "Silver": [453, 653, 582]},
                 material_waste_ratio={"Product": 0., "Aluminum": 0.,
                                       "Glass": 0., "Copper": 0.,
                                       "Insulated cable": 0., "Silicon": 0.4,
                                       "Silver": 0.},
                 recovery_fractions={"Product": np.nan, "Aluminum": 0.92,
                                       "Glass": 0.85, "Copper": 0.72,
                                       "Insulated cable": 1, "Silicon": 0,
                                       "Silver": 0},
                 product_average_wght=0.1,
                 mass_to_function_reg_coeff=0.03,
                 recycling_states=[
                     'Texas', 'Arizona', 'Oregon', 'Oklahoma',
                         'Wisconsin', 'Ohio', 'Kentucky', 'South Carolina'],
                 transportation_cost=0.0314,
                 used_product_substitution_rate=[0.6, 1, 0.8],
                 imperfect_substitution=0,
                 epr_business_model=False,
                 recycling_process={"frelp": False, "asu": False,
                                    "hybrid": False},
                 industrial_symbiosis=False,
                 dynamic_lifetime_model={"Dynamic lifetime": False,
                                         "d_lifetime_intercept": 15.9,
                                         "d_lifetime_reg_coeff": 0.87,
                                         "Seed": False, "Year": 5,
                                         "avg_lifetime": 50},
                 extended_tpb={"Extended tpb": False,
                               "w_convenience": 0.28, "w_knowledge": -0.51,
                               "knowledge_distrib": [0.5, 0.49]},
                 seeding={"Seeding": False,
                          "Year": 10, "number_seed": 50},
                 seeding_recyc={"Seeding": False,
                          "Year": 10, "number_seed": 50, "discount": 0.35}):
        """
        Initiate model
        """
        # Set up variables
        super().__init__(seed=seed)
        self.seed = seed
        att_distrib_param_eol[0] = calibration_n_sensitivity
        att_distrib_param_reuse[0] = calibration_n_sensitivity_2
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.num_consumers = num_consumers
        self.consumers_node_degree = consumers_node_degree
        self.consumers_network_type = consumers_network_type
        
        self.num_recyclers = num_recyclers # 回收商
        self.num_producers = num_producers # 制造商
        self.num_prod_n_recyc = num_recyclers + num_producers # 回收商+制造商
        self.num_sechdstores = num_sechdstores # 二手商
        
        self.prod_n_recyc_node_degree = prod_n_recyc_node_degree
        self.prod_n_recyc_network_type = prod_n_recyc_network_type
        
        self.init_eol_rate = init_eol_rate # dictionary with initial end-of-life (EOL) ratios
        self.init_purchase_choice = init_purchase_rate # dictionary with initial purchase ratios
        
        self.total_number_product = total_number_product # a list for the whole population
        self.copy_total_number_product = self.total_number_product.copy()
        self.mass_to_function_reg_coeff = mass_to_function_reg_coeff
        
        self.clock = 0
        self.iteration = 0
        self.running = True
        self.color_map = []
        self.theory_of_planned_behavior = theory_of_planned_behavior
        self.all_eol_pathways = all_eol_pathways
        self.purchase_options = purchase_choices
        self.avg_failure_rate = failure_rate_alpha
        self.original_num_prod = total_number_product
        self.avg_lifetime = product_lifetime
        
        self.fsthand_mkt_pric = fsthand_mkt_pric                        #### delete
        self.fsthand_mkt_pric_reg_param = fsthand_mkt_pric_reg_param    #### delete
        
        self.average_new_product_price = 5000   # TBD
        self.average_second_hand_price = 2000   # TBD
        self.all_comsumer_income = []
        self.repairability = repairability
        
        self.total_waste = 0
        self.total_yearly_new_products = 0
        self.sold_repaired_waste = 0
        self.past_sold_repaired_waste = 0
        self.repairable_volume_recyclers = 0
        self.consumer_used_product = 0
        self.recycler_repairable_waste = 0
        self.yearly_repaired_waste = 0
        
        self.imperfect_substitution = imperfect_substitution
        perceived_behavioral_control = [np.nan] * len(all_eol_pathways)
        
        # Adjacency matrix of trust network: trust of row index into column
        self.trust_prod = np.asmatrix(np.random.uniform(
            init_trust_boundaries[0], init_trust_boundaries[1],
            (self.num_prod_n_recyc, self.num_prod_n_recyc)))
        np.fill_diagonal(self.trust_prod, 0)
        
        self.social_event_boundaries = social_event_boundaries
        self.trust_threshold = trust_threshold
        self.knowledge_threshold = knowledge_threshold
        self.willingness_threshold = willingness_threshold
        self.willingness = np.asmatrix(np.zeros((self.num_prod_n_recyc, self.num_prod_n_recyc)))
        self.product_mass_fractions = product_mass_fractions
        self.material_waste_ratio = material_waste_ratio
        self.established_scd_mkt = established_scd_mkt
        self.recovery_fractions = recovery_fractions
        self.product_average_wght = product_average_wght
        self.dynamic_product_average_wght = product_average_wght # (kg/fu), MCE: module efficiency
        self.yearly_product_wght = product_average_wght
        self.transportation_cost = transportation_cost # ($/t.km) Transportation cost per unit distance
        self.epr_business_model = epr_business_model
        self.average_landfill_cost = sum(landfill_cost) / len(landfill_cost)
        self.industrial_symbiosis = industrial_symbiosis
        self.installer_recycled_amount = 0
        
        # Change eol_pathways depending on business model
        if self.epr_business_model:
            self.all_eol_pathways["landfill"] = False
            self.industrial_symbiosis = False
        
        # Dynamic lifetime model
        self.dynamic_lifetime_model = dynamic_lifetime_model
        self.extended_tpb = extended_tpb
        self.seeding = seeding
        self.seeding_recyc = seeding_recyc
        self.cost_seeding = 0
        self.product_lifetime = product_lifetime
        self.d_product_lifetimes = []
        self.update_dynamic_lifetime()
        self.original_recycling_cost = original_recycling_cost
        self.recycling_process = recycling_process
        
        self.list_consumer_id = list(range(num_consumers))
        random.shuffle(self.list_consumer_id)
        self.list_consumer_id_seed = list(range(num_consumers))
        random.shuffle(self.list_consumer_id_seed)
        
        # Change recovery fractions and recycling costs depending on recycling process
        self.recycling_process_change()
        self.product_growth = product_growth
        self.growth_threshold = growth_threshold
        
        ####################################################
        #                                                  #
        #                  Building graphs                 #
        #                                                  #
        ####################################################
        
        # Consumer's network
        self.consumer_network = self.init_network(
            network=self.consumers_network_type,
            nodes=self.num_consumers,
            node_degree=self.consumers_node_degree,
            rewiring_prob=rewiring_prob)

        # Manufacturer and Recycler's network
        self.recycler_network = self.init_network(
            network=self.prod_n_recyc_network_type,
            nodes=self.num_prod_n_recyc,
            node_degree=self.prod_n_recyc_node_degree,
            rewiring_prob=rewiring_prob)

        # Second-hand store's network
        self.sechdstore_network = self.init_network(
            network="complete graph",
            nodes=self.num_sechdstores,
            node_degree="NaN",
            rewiring_prob=rewiring_prob)

        #self.network = nx.disjoint_union(self.consumer_network, self.recycler_network)
        self.network = nx.disjoint_union(
            nx.disjoint_union(self.consumer_network, self.recycler_network), 
            self.sechdstore_network)
        self.grid = NetworkGrid(self.network)

        self.all_states = ['Texas', 'California', 'Montana', 'New Mexico',
                      'Arizona', 'Nevada', 'Colorado', 'Oregon', 'Wyoming',
                      'Michigan', 'Minnesota', 'Utah', 'Idaho', 'Kansas',
                      'Nebraska', 'South Dakota', 'Washington',
                      'North Dakota', 'Oklahoma', 'Missouri', 'Florida',
                      'Wisconsin', 'Georgia', 'Illinois', 'Iowa',
                      'New York', 'North Carolina', 'Arkansas', 'Alabama',
                      'Louisiana', 'Mississippi', 'Pennsylvania', 'Ohio',
                      'Virginia', 'Tennessee', 'Kentucky', 'Indiana',
                      'Maine', 'South Carolina', 'West Virginia',
                      'Maryland', 'Massachusetts', 'Vermont',
                      'New Hampshire', 'New Jersey', 'Connecticut',
                      'Delaware', 'Rhode Island']
        self.states = pd.read_csv("StatesAdjacencyMatrix.csv").to_numpy()
        # Compute distances
        self.mean_distance_within_state = np.nanmean(
            np.where(self.states != 0, self.states, np.nan)) / 2
        self.states_graph = nx.from_numpy_array(self.states)
        nodes_states_dic = \
            dict(zip(list(self.states_graph.nodes),
                     list(pd.read_csv("StatesAdjacencyMatrix.csv"))))
        self.states_graph = nx.relabel_nodes(self.states_graph, nodes_states_dic)
        self.recycling_states = recycling_states

        distances_to_recyclers = []
        distances_to_recyclers = self.shortest_paths(
            self.recycling_states, distances_to_recyclers)

        self.mn_mx_av_distance_to_recycler = [ # minimum, maximum, average
            min(distances_to_recyclers), 
            max(distances_to_recyclers),
            sum(distances_to_recyclers) / len(distances_to_recyclers)]
        
        # Compute transportation costs
        self.transportation_cost_rcl = [
            x * self.transportation_cost / 1E3 * self.dynamic_product_average_wght for x in
            self.mn_mx_av_distance_to_recycler]
        
        self.transportation_cost_rpr_ldf = self.mean_distance_within_state * \
            self.transportation_cost / 1E3 * self.dynamic_product_average_wght
        
        # Add transportation costs to pathways' costs
        # Recycling_cost
        self.original_recycling_cost = [sum(x) for x in zip(
            self.original_recycling_cost, self.transportation_cost_rcl)]
        
        original_repairing_cost = [
            x + self.transportation_cost_rpr_ldf for x in original_repairing_cost]
        # Landfilling cost
        landfill_cost = [x + self.transportation_cost_rpr_ldf for x in landfill_cost]

        # self.consumer_agent_set = AgentSet(self)
        # Create agents, G nodes labels are equal to agents' unique_ID
        for node in self.network.nodes():
            #===================== Consumers =====================#
            if node < self.num_consumers:
                a = Consumer(
                    model=self,     # ABM_Model
                    unique_id=node, # agent ID ~ node ID in the network
                    ) 
                self.grid.place_agent(a, node)  # Add the agent to the node
                
            elif node < self.num_recyclers + self.num_consumers:
                #===================== Recyclers =====================#
                b = Recycler(
                    model=self,
                    unique_id=node, 
                    )
                self.grid.place_agent(b, node)

            elif node < self.num_prod_n_recyc + self.num_consumers:
                #===================== Producers =====================#
                c = Manufacturer(
                    model=self,
                    unique_id=node, 
                    )
                self.grid.place_agent(c, node)

            else:
                #================== Second-hand Store ==================#
                d = SecondHandStore(
                    model=self,
                    unique_id=node, 
                    )
                self.grid.place_agent(d, node)

        # Draw initial graph
        # nx.draw(self.G, with_labels=True)
        # plt.show()

        # Defines reporters and set up data collector
        # model_reporters = {
        #     "Year": lambda c: self.report_output("year"),
        #     "Average weight of waste": lambda c: self.report_output("weight"),
        #     "Agents repairing": lambda c: self.count_eol_products("repairing"),
        #     "Agents selling": lambda c: self.count_eol_products("selling"),
        #     "Agents recycling": lambda c: self.count_eol_products("recycling"),
        #     "Agents landfilling": lambda c: self.count_eol_products("landfilling"),
        #     "Agents storing": lambda c: self.count_eol_products("hoarding"),
        #     "Agents buying new": lambda c: self.count_eol_products("buy_new"),
        #     "Agents buying used": lambda c: self.count_eol_products("buy_used"),
        #     "Agents buying certified": lambda c: self.count_eol_products("certified"),
        #     "Total product": lambda c:self.report_output("product_stock"),
        #     "New product": lambda c:self.report_output("product_stock_new"),
        #     "Used product": lambda c:self.report_output("product_stock_used"),
        #     "New product_mass": lambda c:self.report_output("prod_stock_new_mass"),
        #     "Used product_mass": lambda c:self.report_output("prod_stock_used_mass"),
        #     "EoL-repaired": lambda c:self.report_output("product_repaired"),
        #     "EoL-sold": lambda c: self.report_output("product_sold"),
        #     "EoL-recycled": lambda c:self.report_output("product_recycled"),
        #     "EoL-landfilled": lambda c:self.report_output("product_landfilled"),
        #     "EoL-stored": lambda c:self.report_output("product_hoarded"),
        #     "EoL-new repaired weight": lambda c:self.report_output("product_new_repaired"),
        #     "EoL-new sold weight": lambda c:self.report_output("product_new_sold"),
        #     "EoL-new recycled weight": lambda c:self.report_output("product_new_recycled"),
        #     "EoL-new landfilled weight": lambda c:self.report_output("product_new_landfilled"),
        #     "EoL-new stored weight": lambda c:self.report_output("product_new_hoarded"),
        #     "EoL-used repaired weight": lambda c:self.report_output("product_used_repaired"),
        #     "EoL-used sold weight": lambda c:self.report_output("product_used_sold"),
        #     "EoL-used recycled weight": lambda c:self.report_output("product_used_recycled"),
        #     "EoL-used landfilled weight": lambda c:self.report_output("product_used_landfilled"),
        #     "EoL-used stored weight": lambda c:self.report_output("product_used_hoarded"),
        #     "Average landfilling cost": lambda c:self.report_output("average_landfill_cost"),
        #     "Average storing cost": lambda c:self.report_output("average_hoarding_cost"),
        #     "Average recycling cost": lambda c:self.report_output("average_recycling_cost"),
        #     "Average repairing cost": lambda c:self.report_output("average_repairing_cost"),
        #     "Average selling cost": lambda c:self.report_output("average_second_hand_price"),
        #     "Recycled material volume": lambda c:self.report_output("recycled_mat_volume"),
        #     "Recycled material value": lambda c:self.report_output("recycled_mat_value"),
        #     "Producer costs": lambda c:self.report_output("producer_costs"),
        #     "Consumer costs": lambda c:self.report_output("consumer_costs"),
        #     "Recycler costs": lambda c:self.report_output("recycler_costs"),
        #     "Refurbisher costs": lambda c:self.report_output("refurbisher_costs"),
        #     "Refurbisher costs w margins": lambda c:self.report_output("refurbisher_costs_w_margins")}

        # agent_reporters = {
        #     "Year": 
        #         lambda c: self.report_output("year"),
        #     "Number_product_repaired":
        #         lambda a: getattr(a, "number_product_repaired", None),
        #     "Number_product_sold":
        #         lambda a: getattr(a, "number_product_sold", None),
        #     "Number_product_recycled":
        #         lambda a: getattr(a, "number_product_recycled", None),
        #     "Number_product_landfilled":
        #         lambda a: getattr(a, "number_product_landfilled", None),
        #     "Number_product_hoarded":
        #         lambda a: getattr(a, "number_product_hoarded", None),
        #     "Recycling":
        #         lambda a: getattr(a, "EoL_pathway", None),
        #     "Landfilling costs":
        #         lambda a: getattr(a, "landfill_cost", None),
        #     "Storing costs":
        #         lambda a: getattr(a, "hoarding_cost", None),
        #     "Recycling costs":
        #         lambda a: getattr(a, "recycling_cost", None),
        #     "Repairing costs":
        #         lambda a: getattr(a, "repairing_cost", None),
        #     "Selling costs":
        #         lambda a: getattr(a, "scd_hand_price", None),
        #     "Material produced":
        #         lambda a: getattr(a, "material_produced", None),
        #     "Recycled volume":
        #         lambda a: getattr(a, "recycled_material_volume", None),
        #     "Recycled value":
        #         lambda a: getattr(a, "recycled_material_value", None),
        #     "Producer costs":
        #         lambda a: getattr(a, "producer_costs", None),
        #     "Consumer costs":
        #         lambda a: getattr(a, "consumer_costs", None),
        #     "Recycler costs":
        #         lambda a: getattr(a, "recycler_costs", None),
        #     "Refurbisher costs":
        #         lambda a: getattr(a, "refurbisher_costs", None)}

        # self.datacollector = DataCollector(
        #     model_reporters=model_reporters,
        #     agent_reporters=agent_reporters)
        
    def shortest_paths(self, target_states, distances_to_target):
        """
        Compute shortest paths between chosen origin states and targets with
        the Dijkstra algorithm.
        """
        for i in self.states_graph.nodes:
            shortest_paths = []
            for j in target_states:
                shortest_paths.append(
                    nx.shortest_path_length(
                        self.states_graph, 
                        source=i,
                        target=j, 
                        weight='weight',
                        method='dijkstra'))
            shortest_paths_closest_target = min(shortest_paths)
            if shortest_paths_closest_target == 0:
                shortest_paths_closest_target = self.mean_distance_within_state
            distances_to_target.append(shortest_paths_closest_target)
        return distances_to_target

    def init_network(self, network, nodes, node_degree, rewiring_prob):
        """
        Set up model's industrial symbiosis (IS) and consumers networks.
        """
        if network == "small-world":
            return nx.watts_strogatz_graph(
                nodes, node_degree, rewiring_prob, seed=random.seed(self.seed))
        elif network == "complete graph":
            return nx.complete_graph(nodes)
        if network == "random":
            return nx.watts_strogatz_graph(nodes, node_degree, 1)
        elif network == "cycle graph":
            return nx.cycle_graph(nodes)
        elif network == "scale-free graph":
            return nx.powerlaw_cluster_graph(nodes, node_degree, 0.1)
        else:
            return nx.watts_strogatz_graph(nodes, node_degree, rewiring_prob)

    def update_dynamic_lifetime(self):
        """

        """
        if self.dynamic_lifetime_model["Dynamic lifetime"]:
            self.d_product_lifetimes = [
                self.dynamic_lifetime_model["d_lifetime_intercept"] +
                self.dynamic_lifetime_model["d_lifetime_reg_coeff"] *
                x for x in range(len(self.total_number_product) + self.clock
                                 + 1)]
        
        elif self.dynamic_lifetime_model["Seed"]:
            self.d_product_lifetimes = \
                [self.product_lifetime] * \
                (len(self.total_number_product) + self.clock + 1)
            if self.clock >= self.dynamic_lifetime_model["Year"]:
                for i in range(1, self.clock + 2 -
                               self.dynamic_lifetime_model["Year"]):
                    self.d_product_lifetimes[-i] = \
                        self.dynamic_lifetime_model["avg_lifetime"]
        else:
            self.d_product_lifetimes = \
                [self.product_lifetime] * \
                (len(self.total_number_product) + self.clock + 1)

    def waste_generation(self, avg_lifetime, failure_rate, num_product):
        """
        Generate waste, called by consumers and recyclers/refurbishers
        (to get original recycling/repairing amounts).
        """
        correction_year = len(self.total_number_product) - 1
        return [j * (1 - math.e ** (-(((self.clock + (correction_year - z)) /
                               avg_lifetime[z]) ** failure_rate))).real
                for (z, j) in enumerate(num_product)]

    def recycling_process_change(self):
        """
        Compute changes to recycling parameters according to the
        techno-economic analysis of the FRELP, ASU and hybrid recycling
        processes from Heath et al. unpublished techno-economic analysis.
        """
        if self.recycling_process["frelp"]:
            self.recovery_fractions = {
                "Product": np.nan, "Aluminum": 0.994, "Glass": 0.98,
                "Copper": 0.97, "Insulated cable": 1., "Silicon": 0.97,
                "Silver": 0.94}
            self.original_recycling_cost = [0.068, 0.068, 0.068]
            self.industrial_symbiosis = False
        elif self.recycling_process["asu"]:
            self.recovery_fractions = {
                "Product": np.nan, "Aluminum": 0.94, "Glass": 0.99,
                "Copper": 0.83, "Insulated cable": 1., "Silicon": 0.90,
                "Silver": 0.74}
            self.original_recycling_cost = [0.153, 0.153, 0.153]
            self.industrial_symbiosis = False
        elif self.recycling_process["hybrid"]:
            self.recovery_fractions = {
                "Product": np.nan, "Aluminum": 0.994, "Glass": 0.98,
                "Copper": 0.83, "Insulated cable": 1., "Silicon": 0.97,
                "Silver": 0.74}
            self.original_recycling_cost = [0.055, 0.055, 0.055]
            self.industrial_symbiosis = False

    def average_mass_per_function_model(self, product_as_function):
        """
        Compute the weighted average mass of the product's waste volume (in
        fu). The weights are the amount of waste for each year. The weighted
        average mass is returned each time step of the simulation.
        """
        if self.clock <= self.growth_threshold:
            product_growth_rate = self.product_growth[0]
        else:
            product_growth_rate = self.product_growth[1]
        additional_capacity = sum(product_as_function) * product_growth_rate
        product_as_function.append(additional_capacity)
        mass_conversion_coeffs = [
            self.product_average_wght * math.e ** (-self.mass_to_function_reg_coeff * x) 
                for x in range(len(product_as_function))]
        self.yearly_product_wght = mass_conversion_coeffs[-1]
        weighted_average_mass_watt = sum(
            [product_as_function[i] / sum(product_as_function) *
             mass_conversion_coeffs[i] for i
             in range(len(mass_conversion_coeffs))])
        return weighted_average_mass_watt

    def average_price_per_function_model(self):
        """
        Compute the price of first hand products. Price ratio is compared to
        modules of the same year.
        """
        correction_year = len(self.total_number_product)
        self.fsthand_mkt_pric = self.fsthand_mkt_pric_reg_param[0] * math.e **(
                -self.fsthand_mkt_pric_reg_param[1] * (self.clock + correction_year))

    def count_eol_products(self, condition):
        """
        Count adoption in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        for agent in self.schedule.agents:
            if agent.unique_id < self.num_consumers:
                if condition == "repairing" and agent.EoL_pathway == "repair":
                    count += 1
                elif condition == "selling" and agent.EoL_pathway == "sell":
                    count += 1
                elif condition == "recycling" and agent.EoL_pathway == "recycle":
                    count += 1
                elif condition == "landfilling" and agent.EoL_pathway == "landfill":
                    count += 1
                elif condition == "hoarding" and agent.EoL_pathway == "hoard":
                    count += 1
                elif condition == "buy_new" and agent.purchase_choice == "new":
                    count += 1
                elif condition == "buy_used" and agent.purchase_choice == "used":
                    count += 1
                    self.consumer_used_product += 1
                elif condition == "buy_certified" and \
                        agent.purchase_choice == "certified":
                    count += 1
                else:
                    continue
            else:
                continue
        return count

    def report_output(self, condition):
        """
        Count waste streams in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        count2 = 0
        industrial_waste_landfill = 0
        industrial_waste_recycled = 0
        industrial_waste_landfill_mass = 0
        industrial_waste_recycled_mass = 0
        for agent in self.schedule.agents:
            if self.num_consumers + self.num_recyclers <= agent.unique_id < \
                    self.num_consumers + self.num_prod_n_recyc:
                if self.epr_business_model:
                    industrial_waste_recycled += \
                        agent.industrial_waste_generated / self.num_consumers
                    industrial_waste_recycled_mass += \
                        self.yearly_product_wght * \
                        agent.industrial_waste_generated / self.num_consumers
                else:
                    industrial_waste_landfill += \
                        agent.industrial_waste_generated / self.num_consumers
                    industrial_waste_landfill_mass += \
                        self.yearly_product_wght * \
                        agent.industrial_waste_generated / self.num_consumers
        for agent in self.schedule.agents:
            if condition == "product_stock" and agent.unique_id < \
                    self.num_consumers:
                count += sum(agent.number_product_hard_copy)
            elif condition == "product_stock_new" and agent.unique_id < \
                    self.num_consumers:
                count += sum(agent.new_products_hard_copy)
            if condition == "product_stock_used" and agent.unique_id < \
                    self.num_consumers:
                count += sum(agent.used_products_hard_copy)
            elif condition == "prod_stock_new_mass" and agent.unique_id < \
                    self.num_consumers:
                count += agent.new_products_mass
            if condition == "prod_stock_used_mass" and agent.unique_id < \
                    self.num_consumers:
                count += agent.used_products_mass
            elif condition == "product_repaired" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_product_repaired
            elif condition == "product_sold" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_product_sold
                count2 += agent.number_product_sold
                count2 += agent.number_product_repaired
            elif condition == "product_recycled" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_product_recycled
                count += industrial_waste_recycled
            elif condition == "product_landfilled" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_product_landfilled
                count += industrial_waste_landfill
            elif condition == "product_hoarded" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_product_hoarded
            elif condition == "product_new_repaired" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_new_prod_repaired
            elif condition == "product_new_sold" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_new_prod_sold
            elif condition == "product_new_recycled" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_new_prod_recycled
                count += industrial_waste_recycled_mass
            elif condition == "product_new_landfilled" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_new_prod_landfilled
                count += industrial_waste_landfill_mass
            elif condition == "product_new_hoarded" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_new_prod_hoarded
            elif condition == "product_used_repaired" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_used_prod_repaired
            elif condition == "product_used_sold" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_used_prod_sold
            elif condition == "product_used_recycled" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_used_prod_recycled
            elif condition == "product_used_landfilled" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_used_prod_landfilled
            elif condition == "product_used_hoarded" and agent.unique_id < \
                    self.num_consumers:
                count += agent.number_used_prod_hoarded
            elif condition == "consumer_costs" and agent.unique_id < \
                    self.num_consumers:
                count += agent.consumer_costs
            elif condition == "average_landfill_cost" and agent.unique_id < \
                    self.num_consumers:
                count += agent.landfill_cost / self.num_consumers
            elif condition == "average_hoarding_cost" and agent.unique_id < \
                    self.num_consumers:
                count += agent.hoarding_cost / self.num_consumers
            elif condition == "average_recycling_cost" and self.num_consumers\
                    <= agent.unique_id < self.num_consumers + \
                    self.num_recyclers:
                count += agent.recycling_cost / self.num_recyclers
            elif condition == "average_repairing_cost" and self.num_consumers\
                    + self.num_prod_n_recyc <= agent.unique_id:
                count += agent.repairing_cost / self.num_sechdstores
            elif condition == "average_second_hand_price" and \
                    self.num_consumers + self.num_prod_n_recyc <= agent.unique_id:
                count += (-1 * agent.scd_hand_price) / self.num_sechdstores
            elif condition == "year":
                count = 2020 + self.clock
            elif condition == "weight":
                count = self.dynamic_product_average_wght
            elif condition == "recycled_mat_volume" and self.num_consumers + \
                    self.num_recyclers <= agent.unique_id < \
                    self.num_consumers + self.num_prod_n_recyc:
                if not np.isnan(agent.recycled_material_volume):
                    count += agent.recycled_material_volume
            elif condition == "recycled_mat_value" and self.num_consumers + \
                    self.num_recyclers <= agent.unique_id < \
                    self.num_consumers + self.num_prod_n_recyc:
                if not np.isnan(agent.recycled_material_value):
                    count += agent.recycled_material_value
            elif condition == "producer_costs" and self.num_consumers + \
                    self.num_recyclers <= agent.unique_id < \
                    self.num_consumers + self.num_prod_n_recyc:
                count += agent.producer_costs
            elif condition == "recycler_costs" and self.num_consumers <= \
                    agent.unique_id < self.num_consumers + \
                    self.num_recyclers:
                count += agent.recycler_costs
            elif condition == "refurbisher_costs" and self.num_consumers + \
                    self.num_prod_n_recyc <= agent.unique_id:
                count += agent.refurbisher_costs
            elif condition == "refurbisher_costs_w_margins" and self.num_consumers + \
                    self.num_prod_n_recyc <= agent.unique_id:
                count += agent.refurbisher_costs_w_margins
        if condition == "product_sold":
            self.sold_repaired_waste += count2 - self.past_sold_repaired_waste
            self.past_sold_repaired_waste = count2
        return count

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        # update consumer incomes
        self.all_comsumer_income = [
            agent.income for agent in self.agents if isinstance(agent, Consumer)]
            
        self.total_waste = 0
        # self.total_yearly_new_products = 0
        # self.consumer_used_product = 0
        # self.yearly_repaired_waste = 0
        
        # self.dynamic_product_average_wght = \
        #     self.average_mass_per_function_model(self.copy_total_number_product)
        # # Collect data
        # self.datacollector.collect(self)
    
        # # Refers to agent step function
        # self.update_dynamic_lifetime()

        # self.average_price_per_function_model()
        
        # 各个Agent的先后顺序问题
        self.agents_by_type[Consumer].shuffle_do('step')
        self.agents_by_type[Manufacturer].shuffle_do('step')
        self.agents_by_type[SecondHandStore].shuffle_do('step')
        self.agents_by_type[Recycler].shuffle_do('step')

