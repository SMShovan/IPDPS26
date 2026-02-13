import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("escher_vs_thymep_deletevary_temporal.txt", delim_whitespace=True)
    df = df.rename(
        columns={
            "DeletePercentage": "Delete Percentage",
            "TimeSeconds": "Time (seconds)",
        }
    )

    sns.set_context("talk", font_scale=1.0)
    sns.set_palette("colorblind")

    datasets = ["Coauth-DBLP", "Tags-stack-overflow", "com-Orkut", "Threads-stack-overflow", "random"]
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(3, 4))
        sns.lineplot(
            data=df[df["Dataset"] == dataset],
            x="Delete Percentage",
            y="Time (seconds)",
            hue="Mode",
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("Delete Percentage", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.legend(fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(True)
        plt.savefig(f"{dataset}-del-vary-temporal.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
