import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("outputStepPercentage.txt", delim_whitespace=True)

    dataset_order = ["Coauth", "Tags", "Orkut", "Threads", "Random"]
    step_order = ["Construction", "Deletion", "Insertion", "Update"]
    df_pivot = df.pivot(index="Dataset", columns="Steps", values="Percentage").loc[dataset_order, step_order]

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
        }
    )

    ax = df_pivot.plot(kind="bar", stacked=True, figsize=(6, 4), colormap="mako")
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.ylabel("Percentage", fontsize=16)
    plt.xticks(rotation=0)

    legend = plt.legend(
        title="Steps",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.6),
        ncol=2,
        frameon=True,
    )
    legend.get_frame().set_alpha(0.6)

    plt.tight_layout()
    plt.savefig("stacked_percentage.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
