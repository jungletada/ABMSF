import mesa
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from ABM_Model import SmartphoneModel
from ABM_Smartphone import Smartphone
from ABM_Manufacturer import Manufacturer


def plot_line(results_df):    
    g = sns.lineplot(
        data=results_df,
        x="Step",
        y="avg_consumer_income",
    )
    g.figure.set_size_inches(8, 4)
    plot_title = "Average consumer income over time."
    g.set(title=plot_title, ylabel="Income");
    figure = g.get_figure()
    figure.savefig('results/avg_consumer_income.png', dpi=400)


def plot_pathway_action(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["Agents buying new"],
        label="buying new")
    plt.plot(
        results_df["Step"],
        results_df["Agents buying used"],
        label="buying used")
    plt.plot(
        results_df["Step"],
        results_df["Agents repairing"],
        label="repairing")
    plt.plot(
        results_df["Step"],
        results_df["Agents selling"],
        label="selling")
    plt.plot(
        results_df["Step"],
        results_df["Agents recycling"],
        label="recycling")
    plt.plot(
        results_df["Step"],
        results_df["Agents landfilling"],
        label="landfilling")
    plt.plot(
        results_df["Step"],
        results_df["Agents storing"],
        label="storing")
    
    plt.legend()

    plt.xlabel("Step")
    plt.ylabel("Agent Count")
    plt.title("Consumer Actions")
    plt.savefig('results/agents_actions.png')

if __name__ == '__main__':
    csv_file = 'results/output_batch.csv'

    results = mesa.batch_run(
        SmartphoneModel,
        parameters={},
        iterations=1,
        max_steps=60,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_file)
    print(results_df.keys())

    plot_pathway_action(results_df)