import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset_timing(dataset_name: str, output_file: str) -> pd.DataFrame:
    df = pd.read_csv(output_file, delim_whitespace=True)
    df["Dataset"] = dataset_name
    df["Changed Edges"] = df["ChangedEdges"].map({50000: "50K", 100000: "100K", 200000: "200K"})
    df = df.rename(columns={"TimeSeconds": "Time (seconds)"})
    return df[["Dataset", "Changed Edges", "Time (seconds)"]]


def main() -> None:
    frames = [
        load_dataset_timing("Coauth", "outputCoauth.txt"),
        load_dataset_timing("Tags", "outputTags.txt"),
        load_dataset_timing("Orkut", "outputOrkut.txt"),
        load_dataset_timing("Threads", "outputThreads.txt"),
        load_dataset_timing("Random", "outputRandom.txt"),
    ]
    df = pd.concat(frames, ignore_index=True)

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
        }
    )

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=df,
        x="Dataset",
        y="Time (seconds)",
        hue="Changed Edges",
        palette="rocket",
        dodge=True,
        order=["Coauth", "Tags", "Orkut", "Threads", "Random"],
        hue_order=["50K", "100K", "200K"],
    )

    ax.set_axisbelow(True)
    plt.grid(True)
    plt.xlabel("Dataset", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)
    legend = plt.legend(
        title="Changed Edges",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=True,
    )
    legend.get_frame().set_alpha(0.6)
    plt.tight_layout()
    plt.savefig("time_vs_DeltaE.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
