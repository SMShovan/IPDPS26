import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("outputIncidentVertex.txt", delim_whitespace=True)
    df = df.rename(columns={"TimeMilliseconds": "Time (milliseconds)", "Modification": "Random length"})

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=df,
        x="Dataset",
        y="Time (milliseconds)",
        hue="Random length",
        palette="magma",
        dodge=True,
        order=["Coauth", "Tags", "Orkut", "Threads", "Random"],
        hue_order=["50K", "100K", "200K"],
    )

    plt.xticks(rotation=0)
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.xlabel("Dataset", fontsize=16)
    plt.ylabel("Time (milliseconds)", fontsize=16)
    plt.legend(title="Modification", loc="upper left")
    plt.tight_layout()
    plt.savefig("time_vs_incident.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
