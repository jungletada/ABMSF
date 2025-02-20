import mesa
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from ABM_Model import SmartphoneModel
from ABM_Smartphone import Smartphone
from ABM_Manufacturer import Manufacturer

FIG_DPI = 500

def plot_consumer_income(results_df):    
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["avg_consumer_income"],
        label="buying new")
    plt.title("Average Consumer Income Over Time")
    plt.xlabel("Months")
    plt.ylabel("Income")
    plt.savefig('results/avg_consumer_income.png', dpi=FIG_DPI)


def plot_pathway_action(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["consumer_buying new"],
        label="buying new")
    plt.plot(
        results_df["Step"],
        results_df["consumer_buying used"],
        label="buying used")
    plt.plot(
        results_df["Step"],
        results_df["consumer_repairing"],
        label="repairing")
    plt.plot(
        results_df["Step"],
        results_df["consumer_selling"],
        label="selling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_recycling"],
        label="recycling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_landfilling"],
        label="landfilling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_storing"],
        label="storing")
    
    plt.legend()

    plt.xlabel("Step")
    plt.ylabel("Agent Count")
    plt.title("Consumer Actions")
    plt.savefig('results/agents_actions.png', dpi=FIG_DPI)


def plot_product(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["avg_new_product_price"],
        label="new")
    plt.plot(
        results_df["Step"],
        results_df["avg_used_product_price"],
        label="used")
    plt.title("Average Product Price Over Time")
    plt.xlabel("Months")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig('results/avg_product_price.png', dpi=FIG_DPI)


if __name__ == '__main__':
    csv_file = 'results/output_batch.csv'
    results = mesa.batch_run(
        SmartphoneModel,
        parameters={},
        iterations=1,
        max_steps=120,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_file)
    print(results_df.keys())
    plot_consumer_income(results_df)
    plot_pathway_action(results_df)
    plot_product(results_df)