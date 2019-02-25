import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as plot
import numpy as np
from utils import read_data_from_raw

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)
BASE_DIR = "/home/dongki/research/lids/git/2018-hierarchical-teach/tmp/logs/"
session_freq = 600


def read_data(method, custom_param):
    seed_n = np.arange(10)
    seed_n = [2, 3, 4, 5, 6, 7, 8, 9]

    data_n = []
    for seed_i, seed in enumerate(seed_n):
        path = custom_param["base_path"] + "env::simple_spread_flip_seed::" + str(seed) + "_comment::" + method + "_log"

        # Read data
        data = read_data_from_raw(path, key=custom_param["key"], index=-1)
        if seed_i == 0:
            time = read_data_from_raw(path, key=custom_param["key"], index=5)  # Only need to read time first time
            time = np.asarray(time)

        # Smooth data
        # data = moving_average(data)

        data_n.append(data)

    data_n = np.transpose(np.stack(data_n, axis=1))

    return data_n, time


def vis_experiment(custom_param):
    # Read data
    data_n = []
    time_n = []
    for method in custom_param["method_n"]:
        data, time = read_data(method=method, custom_param=custom_param)
        data_n.append(data)
        time_n.append(time)
        
    # Visualize
    fig, ax = plt.subplots()
    for data, time, legend in zip(data_n, time_n, custom_param["legend_n"]):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        error = (mean - std, mean + std)
    
        ax.fill_between(time, error[0], error[1], alpha=0.2)
        ax.plot(time, mean, label=legend)
        ax.margins(x=0)

    plt.text(x=400., y=-0.9, s=r'\textbf{Before Distill}', fontsize=14)
    plt.annotate(s='', xy=(3000, -1.), xytext=(0., -1.), arrowprops=dict(arrowstyle='<->'))

    plt.text(x=5500., y=-0.9, s=r'\textbf{After Distill}', fontsize=14)
    plt.annotate(s='', xy=(3000, -1.), xytext=(10000., -1.), arrowprops=dict(arrowstyle='<->'))

    plt.ylim([-5., -0.5])
    plt.xlabel(r'\textbf{Train Episode}', size=14)
    plt.ylabel(r'\textbf{Evaluation Reward}', size=14)
    plt.title(r'\textbf{' + custom_param["title"] + '}', size=15)
    
    ax.legend(frameon=True, framealpha=0.8)
    # legend = plt.legend(
    #     bbox_to_anchor=(0., 1.07, 1., .102), 
    #     loc=3, 
    #     ncol=2, 
    #     mode="expand", 
    #     borderaxespad=0.)

    # TODO Save figure instead showing
    plt.show()


def main(args):
    if args.plot_option == "spread_2_agent":
        custom_param = {}
        custom_param["base_path"] = "data/simple_spread_flip/"
        custom_param["key"] = "Test Episode"
        custom_param["method_n"] = [
            "no_distill",
            "distill_all",
            "distill_actor_only",
            "distill_critic_only"]
        custom_param["legend_n"] = [
            "No Distill",
            "Distill (All)",
            "distill (Actor Only)",
            "Distill (Critic Only)"]
        custom_param["title"] = "Comparisons in Spread Domain (2 Agent)"

        vis_experiment(custom_param=custom_param)

    else:
        raise ValueError("Invalid option")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--plot-option", type=str, required=True,
        choices=["spread_2_agent"],
        help="Experiment to plot")

    args = parser.parse_args()

    main(args=args)
