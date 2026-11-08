"""plot.py - makes graphs from results/results.csv into results/graphs/

    python3 plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/results.csv")
df = df[df.image == "synthetic"]
os.makedirs("results/graphs", exist_ok=True)

# if a test was run more than once, keep the latest result
keys = ["mode", "filter", "width", "threads", "schedule"]
df = df.drop_duplicates(keys, keep="last")

par = df[(df["mode"] == "parallel") & (df.schedule == "static")].copy()
par["speedup"] = pd.to_numeric(par.speedup, errors="coerce")
par["efficiency"] = pd.to_numeric(par.efficiency, errors="coerce")
size = par.width.max()                      # graphs 1-3 use the biggest image
big = par[par.width == size]
threads = sorted(par.threads.unique())


def finish(ax, title, ylabel, name):
    ax.set_title(title)
    ax.set_xlabel("threads")
    ax.set_ylabel(ylabel)
    ax.set_xticks(threads)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig("results/graphs/" + name, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved results/graphs/" + name)


# 1. execution time vs threads
fig, ax = plt.subplots()
for f, g in big.groupby("filter"):
    ax.plot(g.threads, g.time_s, marker="o", label=f)
finish(ax, f"Execution time ({size}x{size})", "time (s)", "time_vs_threads.png")

# 2. speedup vs threads
fig, ax = plt.subplots()
ax.plot(threads, threads, "k--", label="ideal")
for f, g in big.groupby("filter"):
    ax.plot(g.threads, g.speedup, marker="o", label=f)
finish(ax, f"Speedup ({size}x{size})", "speedup = T_serial / T_parallel",
       "speedup_vs_threads.png")

# 3. efficiency vs threads
fig, ax = plt.subplots()
for f, g in big.groupby("filter"):
    ax.plot(g.threads, g.efficiency * 100, marker="o", label=f)
finish(ax, f"Efficiency ({size}x{size})", "efficiency (%)",
       "efficiency_vs_threads.png")

# 4. speedup vs image size (8 threads)
p = 8 if 8 in threads else max(threads)
fig, ax = plt.subplots()
for f, g in par[par.threads == p].groupby("filter"):
    ax.plot(g.width, g.speedup, marker="o", label=f)
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(par.width.unique()))
ax.set_xticklabels(sorted(par.width.unique()))
ax.set_title(f"Speedup vs image size ({p} threads)")
ax.set_xlabel("image size (N x N)")
ax.set_ylabel("speedup")
ax.grid(alpha=0.3)
ax.legend()
plt.savefig("results/graphs/speedup_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved results/graphs/speedup_vs_size.png")

# 5. schedules: static vs dynamic vs guided
sch = df[(df["mode"] == "parallel") & (df.width == 4096)].copy()
sch = sch[sch.threads.isin(sch[sch.schedule != "static"].threads.unique())]
if (sch.schedule != "static").any():
    sch["speedup"] = pd.to_numeric(sch.speedup, errors="coerce")
    table = sch.pivot_table(index="threads", columns="schedule",
                            values="speedup", aggfunc="mean")
    table.plot(kind="bar", rot=0)
    plt.title("Loop schedules (4096x4096, average over filters)")
    plt.ylabel("speedup")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig("results/graphs/schedules.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved results/graphs/schedules.png")
