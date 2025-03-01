import os
import logging
import mesa
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter

from ABM_Model import AgentBasedModel
from ABM_Smartphone import Smartphone
from ABM_Manufacturer import Manufacturer

from custom_logger import setup_logger
from utils.plot_utils import *


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    csv_file = 'results/output_batch.csv'
    results = mesa.batch_run(
        AgentBasedModel,
        parameters={},
        iterations=1,
        max_steps=120,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_file)

    setup_logger(filename='logs/ABM.log')
    logging.info(results_df.keys())

    plot_consumer_income(results_df)
    plot_buying_action(results_df)
    plot_eol_action(results_df)
    plot_product_price(results_df)
    plot_price2income(results_df)

    total_new = results_df["consumer_buying_new"].sum()
    total_used = results_df["consumer_buying_used"].sum()
    buying_array = np.array([total_new, total_used])
    all_num_buying = buying_array.sum()
    ratio_buying = buying_array / all_num_buying * 100.
    logging.info(f'Consumers buying action={ratio_buying}')
    
    num_repairing = results_df["consumer_repairing"].sum()
    num_selling = results_df["consumer_selling"].sum()
    num_recycling = results_df["consumer_recycling"].sum()
    num_landfilling = results_df["consumer_landfilling"].sum()
    num_storing = results_df["consumer_storing"].sum()
    eol_array = np.array([num_repairing, num_selling, num_recycling, num_landfilling, num_storing])
    all_num_eol = eol_array.sum()
    ratio_eol = eol_array / all_num_eol * 100.
    logging.info(f'Consumers EoL actions={ratio_eol}')