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

    plt.xticks(rotation=0)
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.legend(title="Changed Edges", loc="upper left")
    plt.tight_layout()
    plt.savefig("time_vs_DeltaE.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
