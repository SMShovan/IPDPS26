import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    df = pd.read_csv("cuda_thyme_speedup_timing.txt", delim_whitespace=True)
    df = df.rename(
        columns={
            "ChangedEdges": "Changed Edges",
            "TimeSeconds": "Time (seconds)",
        }
    )

    sns.set_context("talk", font_scale=0.8)
    sns.set_style("whitegrid")

    datasets = ["Coauth", "Tags", "Orkut", "Threads", "Random"]
    changed_edges = ["50K", "100K", "200K"]

    speedup_records = []
    for dataset in datasets:
        for ce in changed_edges:
            dynamic_time = df[
                (df["Dataset"] == dataset)
                & (df["Changed Edges"] == ce)
                & (df["Mode"] == "dynamicCUDA")
            ]["Time (seconds)"].values[0]
            baseline_time = df[
                (df["Dataset"] == dataset)
                & (df["Changed Edges"] == ce)
                & (df["Mode"] == "baselineCUDA")
            ]["Time (seconds)"].values[0]
            speedup_records.append(
                {
                    "Dataset": dataset,
                    "Changed Edges": ce,
                    "Speedup": baseline_time / dynamic_time,
                }
            )

    speedup_df = pd.DataFrame(speedup_records)
    pivot_df = speedup_df.pivot(index="Dataset", columns="Changed Edges", values="Speedup")
    pivot_df = pivot_df[changed_edges]

    fig, ax = plt.subplots(figsize=(6, 4))
    pivot_df.plot(kind="bar", ax=ax, width=0.9, color=sns.color_palette("tab20"))

    ax.set_ylabel("Speedup (THyMe + GPU / CHAI)")
    ax.set_xlabel("Dataset")
    ax.set_title("Speedup across Datasets and Changed Edges")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    plt.xticks(rotation=0, ha="right")
    plt.legend(title="Changed Edges")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=12)

    plt.tight_layout()
    plt.savefig("SpeedupCUDATHyMe.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
