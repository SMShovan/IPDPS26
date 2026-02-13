import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("outputHyperedgeCardinality.txt", delim_whitespace=True)
    df = df.rename(columns={"TimeSeconds": "Time (seconds)"})
    df["Random length"] = df["Cardinality"].astype(str)

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16
    })

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=df,
        x="Dataset",
        y="Time (seconds)",
        hue="Random length",
        palette="viridis",
        dodge=True,
        order=["Coauth", "Tags", "Orkut", "Threads", "Random"],
        hue_order=["50", "100", "200"],
    )

    ax.set_axisbelow(True)
    plt.grid(True)
    plt.xlabel("Dataset", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)
    legend = plt.legend(
        title="Hyperedge Cardinality",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=True,
    )
    legend.get_frame().set_alpha(0.6)
    plt.tight_layout()
    plt.savefig("time_vs_length.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
