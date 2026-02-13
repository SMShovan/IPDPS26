import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("outputHyperedgeSize.txt", delim_whitespace=True)
    df = df.rename(columns={"GraphNodes": "Graph Nodes", "TimeSeconds": "Time (seconds)"})

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="Graph Nodes", y="Time (seconds)", palette="rocket")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

    ax.set_xlabel("Hyperedge counts")
    ax.set_ylabel("Time (seconds)")
    plt.title("", fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("time_vs_vertexcount.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
