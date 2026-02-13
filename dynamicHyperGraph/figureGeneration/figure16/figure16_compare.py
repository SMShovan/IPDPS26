import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("hornet_vs_escher_time_memory.txt", delim_whitespace=True)
    df = df.rename(columns={"TimeSeconds": "Time (seconds)", "MemoryMB": "Memory (mb)", "STD": "STD (on Avg)"})

    pivot_time = df.pivot(index="STD (on Avg)", columns="Mode", values="Time (seconds)")
    pivot_mem = df.pivot(index="STD (on Avg)", columns="Mode", values="Memory (mb)")

    common_time = pivot_time.dropna(subset=["Hornet", "ESCHER"]).sort_index()
    common_mem = pivot_mem.dropna(subset=["Hornet", "ESCHER"]).sort_index()

    common_index = common_time.index.intersection(common_mem.index)
    common_time = common_time.loc[common_index]
    common_mem = common_mem.loc[common_index]

    ratio_time = (common_time["Hornet"] / common_time["ESCHER"]).rename("Time Ratio")
    ratio_mem = (common_mem["Hornet"] / common_mem["ESCHER"]).rename("Memory Ratio")

    plt.figure(figsize=(5, 4))
    plt.plot(common_index, ratio_time.values, marker="o", linewidth=2, label="Time Ratio (Hornet/ESCHER)")
    plt.plot(common_index, ratio_mem.values, marker="s", linewidth=2, label="Memory Ratio (Hornet/ESCHER)")
    plt.axhline(1, linestyle="--", linewidth=1, color="gray")
    plt.title("")
    plt.xlabel("STD (on Avg)")
    plt.ylabel("Speedup")
    plt.xticks(common_index)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="center", bbox_to_anchor=(0.5, 0.45), frameon=True, borderpad=0.4, handlelength=2)
    plt.tight_layout()
    out_path = "hornet_proposed_ratio_line.png"
    plt.savefig(out_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
