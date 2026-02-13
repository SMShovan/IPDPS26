import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


def main() -> None:
    df = pd.read_csv("figure11_types_speedup.txt", delim_whitespace=True)
    df = df.rename(
        columns={
            "ChangedEdges": "Changed Edges",
            "Type1Speedup": "Type 1 Speedup",
            "Type2Speedup": "Type 2 Speedup",
            "Type3Speedup": "Type 3 Speedup",
        }
    )

    df_long = df.melt(
        id_vars=["Dataset", "Changed Edges"],
        value_vars=["Type 1 Speedup", "Type 2 Speedup", "Type 3 Speedup"],
        var_name="Type",
        value_name="Speedup",
    )
    df_long["Type"] = df_long["Type"].str.replace(" Speedup", "", regex=False)

    datasets = ["Coauth", "Tags", "Orkut", "Threads", "Random"]
    changed_edges = ["50K", "100K", "200K"]
    types = ["Type 1", "Type 2", "Type 3"]

    df_long["Dataset"] = pd.Categorical(df_long["Dataset"], categories=datasets, ordered=True)
    df_long["Changed Edges"] = pd.Categorical(df_long["Changed Edges"], categories=changed_edges, ordered=True)
    df_long["Type"] = pd.Categorical(df_long["Type"], categories=types, ordered=True)
    df_long = df_long.sort_values(["Dataset", "Changed Edges", "Type"])

    n_ce = len(changed_edges)
    n_t = len(types)
    w = 0.08
    gap_ce = 0.06
    total_width = n_ce * (n_t * w) + (n_ce - 1) * gap_ce

    x_centers = {ds: i for i, ds in enumerate(datasets)}
    left_offset = -total_width / 2 + w / 2

    ce_colors = {ce: c for ce, c in zip(changed_edges, sns.color_palette("Pastel1", n_ce))}
    type_hatches = {"Type 1": "xxx", "Type 2": "ooo", "Type 3": "|||"}

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")

    for ds in datasets:
        x0 = x_centers[ds] + left_offset
        for j, ce in enumerate(changed_edges):
            cluster_start = x0 + j * (n_t * w + gap_ce)
            for k, t in enumerate(types):
                bar_x = cluster_start + k * w
                val = df_long.loc[
                    (df_long["Dataset"] == ds) & (df_long["Changed Edges"] == ce) & (df_long["Type"] == t), "Speedup"
                ].values[0]

                ax.bar(bar_x, val, width=w, color=ce_colors[ce], edgecolor="black", linewidth=0.6, hatch=type_hatches[t])

    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_xticks([x_centers[ds] for ds in datasets], labels=datasets)
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("Speedup", fontsize=16)

    ce_handles = [Patch(facecolor=ce_colors[ce], edgecolor="black", label=ce) for ce in changed_edges]
    legend_ce = ax.legend(
        handles=ce_handles,
        title="Changed Edges",
        loc="upper left",
        bbox_to_anchor=(0.009, -0.1),
        ncol=len(changed_edges),
        frameon=True,
    )
    legend_ce.get_frame().set_alpha(0.6)
    ax.add_artist(legend_ce)

    type_handles = [Patch(facecolor="white", edgecolor="black", hatch=type_hatches[t], label=t) for t in types]
    legend_t = ax.legend(
        handles=type_handles,
        title="Type",
        loc="upper right",
        bbox_to_anchor=(0.99, -0.1),
        ncol=len(types),
        frameon=True,
    )
    legend_t.get_frame().set_alpha(0.6)

    plt.tight_layout()
    plt.savefig("speedup_vs_deltaE_types_nested.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
