import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("escher_vs_mochy_e_timing.txt", delim_whitespace=True)
    sns.set_context("talk", font_scale=0.7)

    datasets = ["Coauth-DBLP", "Tags-stack-overflow", "com-Orkut", "Threads-stack-overflow", "Random"]

    for dataset in datasets:
        subset = df[df["Dataset"] == dataset]
        escher_data = subset[subset["Mode"] == "ESCHER"]
        mochy_data = subset[subset["Mode"] == "MoCHy"]

        plt.figure(figsize=(2, 3))
        plt.plot(escher_data["ChangedEdges"], escher_data["TimeSeconds"], label="ESCHER", marker="o")
        plt.plot(mochy_data["ChangedEdges"], mochy_data["TimeSeconds"], label="MoCHy", marker="o")

        plt.xlabel("Changed Edges")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.gca().set_axisbelow(True)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{dataset}_base_vs_DeltaE.pdf", format="pdf")
        plt.show()


if __name__ == "__main__":
    main()
