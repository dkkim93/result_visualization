import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as plot
import numpy as np
from utils import read_data_from_raw, moving_average

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)


def read_data(method, custom_param):
    seed_n = np.arange(10)
    data_n = []
    for seed_i, seed in enumerate(seed_n):
        path = \
            custom_param["base_path"] + \
            "env::" + custom_param["env"] + \
            "_seed::" + str(seed) + "_comment::" + method + "_log"

        # Read data
        data = read_data_from_raw(path, key=custom_param["key"], index=-1)
        if seed_i == 0:
            time = read_data_from_raw(path, key=custom_param["key"], index=5)  # Only need to read time first time
            time = np.asarray(time)

        # Smooth data
        data_smooth = moving_average(data, periods=5)
        data[len(data) - len(data_smooth):] = data_smooth

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

    plt.ylim([custom_param["ylim_min"], custom_param["ylim_max"]])
    plt.xlabel(r'\textbf{Train Episode}', size=14)
    plt.ylabel(r'\textbf{Evaluation Reward}', size=14)
    plt.title(r'\textbf{' + custom_param["title"] + '}', size=15)
    ax.legend(frameon=True, framealpha=0.8)

    # Draw arrow 
    plt.text(
        x=custom_param["text_before_at"],
        y=custom_param["arrow_at"] + 0.1, 
        s=r'\textbf{Before Distill}', fontsize=14)
    plt.annotate(
        s='', 
        xy=(custom_param["distill_at"], custom_param["arrow_at"]), 
        xytext=(0., custom_param["arrow_at"]), arrowprops=dict(arrowstyle='<->'))

    plt.text(
        x=custom_param["text_after_at"],
        y=custom_param["arrow_at"] + 0.1, 
        s=r'\textbf{After Distill}', fontsize=14)
    plt.annotate(
        s='', 
        xy=(custom_param["distill_at"], custom_param["arrow_at"]), 
        xytext=(time[-1], custom_param["arrow_at"]), arrowprops=dict(arrowstyle='<->'))

    plt.savefig(custom_param["save_name"], bbox_incehes="tight")


def main(args):
    if args.plot_option == "spread_2_agent":
        custom_param = {}
        custom_param["base_path"] = "data/simple_spread_flip/"
        custom_param["env"] = "simple_spread_flip"
        custom_param["key"] = "Test Episode"
        custom_param["method_n"] = [
            "distill_all",
            "distill_critic_only",
            "distill_actor_only",
            "no_distill"]
        custom_param["legend_n"] = [
            "Distill (All)",
            "Distill (Critic Only)",
            "distill (Actor Only)",
            "No Distill"]
        custom_param["distill_at"] = 3000
        custom_param["arrow_at"] = -1.05
        custom_param["text_before_at"] = 350
        custom_param["text_after_at"] = 5500
        custom_param["title"] = "Comparisons in Spread Domain (2 Agent)"
        custom_param["ylim_min"] = -3.5
        custom_param["ylim_max"] = -0.7
        custom_param["save_name"] = "spread_two_agent.png"

        vis_experiment(custom_param=custom_param)

    elif args.plot_option == "spread_3_agent":
        custom_param = {}
        custom_param["base_path"] = "data/simple_spread_flip_3/"
        custom_param["env"] = "simple_spread_flip_3"
        custom_param["key"] = "Test Episode"
        custom_param["method_n"] = [
            "distill_all",
            "distill_actor_only",
            "distill_critic_only",
            "no_distill"]
        custom_param["legend_n"] = [
            "Distill (All)",
            "distill (Actor Only)",
            "Distill (Critic Only)",
            "No Distill"]
        custom_param["distill_at"] = 4000
        custom_param["text_before_at"] = 850
        custom_param["text_after_at"] = 6000
        custom_param["arrow_at"] = -1.25
        custom_param["title"] = "Comparisons in Spread Domain (3 Agent)"
        custom_param["ylim_min"] = -3.5
        custom_param["ylim_max"] = -1.0
        custom_param["save_name"] = "spread_three_agent.png"

        vis_experiment(custom_param=custom_param)

    elif args.plot_option == "spread_4_agent":
        custom_param = {}
        custom_param["base_path"] = "data/simple_spread_flip_4/"
        custom_param["env"] = "simple_spread_flip_4"
        custom_param["key"] = "Test Episode"
        custom_param["method_n"] = [
            "distill_all",
            "distill_critic_only",
            "distill_actor_only",
            "no_distill"]
        custom_param["legend_n"] = [
            "Distill (All)",
            "Distill (Critic Only)",
            "distill (Actor Only)",
            "No Distill"]
        custom_param["distill_at"] = 5000
        custom_param["arrow_at"] = -1.4
        custom_param["title"] = "Comparisons in Spread Domain (4 Agent)"
        custom_param["ylim_min"] = -4.0
        custom_param["ylim_max"] = -1.0
        custom_param["save_name"] = "spread_four_agent.png"

        vis_experiment(custom_param=custom_param)
    else:
        raise ValueError("Invalid option")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--plot-option", type=str, required=True,
        choices=["spread_2_agent", "spread_3_agent", "spread_4_agent"],
        help="Experiment to plot")

    args = parser.parse_args()

    main(args=args)
