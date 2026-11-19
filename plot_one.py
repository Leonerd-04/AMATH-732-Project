import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
from matplotlib.patches import Circle

# Must match simulation
R_right = 0.5
R_left = 0.5
K_NN = 8
LINK_THR = 0.05

# ============================================================
# LOAD DATA (once)
# ============================================================
print("Loading data...")
df = pd.read_csv("results/individual_infections.csv")
df["sign"] = np.sign(df["infection"]).astype(int)
df["is_infected"] = df["sign"] != 0

times = np.sort(df["time"].unique())
numparts = df["agent_num"].nunique()
groups = {t: d for t, d in df.groupby("time", sort=True)}

sign_mat = df.pivot(index="time", columns="agent_num", values="sign").to_numpy()
x_mat    = df.pivot(index="time", columns="agent_num", values="x").to_numpy()
y_mat    = df.pivot(index="time", columns="agent_num", values="y").to_numpy()
n_t, n_a = sign_mat.shape

side0 = np.where(x_mat[0] > 0, "right", "left")
cx_per_agent = np.where(side0 == "right", R_right, -R_left)
theta_mat = np.arctan2(y_mat, x_mat - cx_per_agent[None, :])

pos_count = (sign_mat == 1).sum(axis=1)
neg_count = (sign_mat == -1).sum(axis=1)
total_count = pos_count + neg_count

OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)
print("Data loaded.")


# ============================================================
# PLOT 1: Spatial snapshots (viridis)
# ============================================================
print("1/6: spatial snapshots...")
snap_idx = [0, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
fig, axes = plt.subplots(1, 5, figsize=(19, 4))
for ax, t in zip(axes, times[snap_idx]):
    d = groups[t]
    und, com = d[d["sign"] == 0], d[d["sign"] != 0]
    ax.scatter(und["x"], und["y"], color="black", s=3, alpha=0.4)
    sc = ax.scatter(com["x"], com["y"], c=com["infection"],
                    cmap="viridis", vmin=-1, vmax=1, s=15)
    ax.add_patch(Circle((R_right, 0), R_right, fill=False, edgecolor="grey", linestyle="--", linewidth=0.6, alpha=0.6))
    ax.add_patch(Circle((-R_left, 0), R_left, fill=False, edgecolor="grey", linestyle="--", linewidth=0.6, alpha=0.6))
    ax.set_title(f"t = {t:.2f}"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
fig.colorbar(sc, ax=axes, label="Opinion (-1 … +1)")
fig.suptitle("Spatial Spread of Opinions")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/spatial_snapshots.png", dpi=200); plt.close()


# ============================================================
# PLOT 2: Distribution (single colour)
# ============================================================
print("2/6: distribution...")
snap_idx = [0, len(times)//4, len(times)//2, len(times)-1]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
bin_edges = np.linspace(-1, 1, 41)
for ax, t in zip(axes.flatten(), times[snap_idx]):
    com = groups[t].loc[groups[t]["sign"] != 0, "infection"].to_numpy()
    ax.hist(com, bins=bin_edges, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(f"t = {t:.2f}"); ax.set_xlabel("Opinion value"); ax.set_ylabel("Count")
    ax.set_xlim(-1, 1)
fig.suptitle("Distribution of Opinion Strength")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/distribution.png", dpi=200); plt.close()


# ============================================================
# PLOT 4: Phase portrait
# ============================================================
print("4/6: phase portrait...")
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(pos_count, neg_count, color="steelblue", linewidth=2)
ax.scatter([pos_count[0]], [neg_count[0]], color="green", s=70, zorder=5, label="start")
ax.scatter([pos_count[-1]], [neg_count[-1]], color="red", s=70, zorder=5, label="end")
if n_t > 1:
    ax.annotate("", xy=(pos_count[-1], neg_count[-1]), xytext=(pos_count[-2], neg_count[-2]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
for it in range(0, n_t, max(1, n_t // 10)):
    ax.scatter([pos_count[it]], [neg_count[it]], color="steelblue", s=10, zorder=4)
ax.set_xlabel("N₊"); ax.set_ylabel("N₋"); ax.set_title("Phase portrait: (N₊, N₋)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/phase_portrait.png", dpi=200); plt.close()


# ============================================================
# ECHO-SILO DIAGNOSTICS
# ============================================================

def knn_segregation(pos, signs, k=K_NN):
    n = len(pos)
    if n < 2: return np.zeros(n), np.zeros(n)
    k = min(k, n - 1)
    _, idx = cKDTree(pos).query(pos, k=k + 1)
    nn_sig = signs[idx[:, 1:]]
    return nn_sig.mean(axis=1), (nn_sig == signs[:, None]).mean(axis=1)

def angular_boundaries(theta, signs, n_bins=36):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    bin_sign = np.zeros(n_bins)
    for ib in range(n_bins):
        sel = bin_idx == ib
        if sel.any():
            net = (signs[sel] == 1).sum() - (signs[sel] == -1).sum()
            bin_sign[ib] = 1 if net > 0 else (-1 if net < 0 else 0)
    trans = 0
    for i in range(n_bins):
        a, b = bin_sign[i], bin_sign[(i + 1) % n_bins]
        if a != 0 and b != 0 and a != b: trans += 1
    return trans, bin_sign

print("Computing echo-silo diagnostics...")
same_frac_pos = np.full(n_t, np.nan); same_frac_neg = np.full(n_t, np.nan)
excess_pos = np.full(n_t, np.nan);    excess_neg = np.full(n_t, np.nan)
interface_count = np.zeros(n_t, dtype=int); same_close_count = np.zeros(n_t, dtype=int)

for it in range(n_t):
    s = sign_mat[it]; xy = np.column_stack([x_mat[it], y_mat[it]])
    comm = s != 0; n_pos = int((s==1).sum()); n_neg = int((s==-1).sum()); n_c = n_pos+n_neg
    if n_c < 2: continue
    pos_c, sign_c = xy[comm], s[comm]
    _, same_frac = knn_segregation(pos_c, sign_c, K_NN)
    glob_pos, glob_neg = n_pos/n_c, n_neg/n_c
    if n_pos > 0:
        m = sign_c == 1; same_frac_pos[it] = same_frac[m].mean(); excess_pos[it] = same_frac[m].mean() - glob_pos
    if n_neg > 0:
        m = sign_c == -1; same_frac_neg[it] = same_frac[m].mean(); excess_neg[it] = same_frac[m].mean() - glob_neg
    pairs = cKDTree(pos_c).query_pairs(r=LINK_THR)
    if pairs:
        i_arr, j_arr = map(np.array, zip(*pairs))
        same = int((sign_c[i_arr] == sign_c[j_arr]).sum())
        same_close_count[it] = same; interface_count[it] = len(i_arr) - same
print("Diagnostics computed.")


# ============================================================
# PLOT 5: Segregation time series
# ============================================================
print("5/6: segregation time series...")

glob_pos_arr = (
    (sign_mat == 1).sum(axis=1) /
    np.maximum((sign_mat != 0).sum(axis=1), 1)
)

# Normalize cross-opinion pairs
cross_pair_fraction = np.full(n_t, np.nan)

for it in range(n_t):
    total_pairs = interface_count[it] + same_close_count[it]

    if total_pairs > 0:
        cross_pair_fraction[it] = interface_count[it] / total_pairs


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ------------------------------------------------------------
# (a) k-NN segregation
# ------------------------------------------------------------
ax = axes[0]

ax.plot(times, same_frac_pos, color="red",
        label="+1 same-sign NN")

ax.plot(times, same_frac_neg, color="blue",
        label="-1 same-sign NN")

ax.plot(times, glob_pos_arr, color="red",
        linestyle=":", alpha=0.5,
        label="random +1")

ax.plot(times, 1-glob_pos_arr, color="blue",
        linestyle=":", alpha=0.5,
        label="random -1")

ax.set_ylabel("Same-sign NN fraction")
ax.set_xlabel("Time")
ax.set_title("(a) k-NN segregation")
ax.set_ylim(0, 1)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)


# ------------------------------------------------------------
# (b) Segregation excess
# ------------------------------------------------------------
ax = axes[1]

ax.plot(times, excess_pos, color="red",
        lw=2, label="+1 excess")

ax.plot(times, excess_neg, color="blue",
        lw=2, label="-1 excess")

ax.axhline(0, color="grey", ls="--", lw=0.8)
ax.axhline(0.2, color="orange", ls=":", lw=0.8)
ax.axhline(-0.2, color="orange", ls=":", lw=0.8)

ax.set_ylabel("Observed - random")
ax.set_xlabel("Time")
ax.set_title("(b) Segregation excess")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)


# ------------------------------------------------------------
# (c) Normalized cross-opinion pairs
# ------------------------------------------------------------
ax = axes[2]

ax.plot(times, cross_pair_fraction,
        color="darkorange", lw=2,
        label="Cross-opinion pair fraction")

ax.axhline(0.5, color="grey",
           linestyle="--", lw=0.8,
           label="50% reference")

ax.set_ylabel("Fraction of cross-opinion pairs")
ax.set_xlabel("Time")
ax.set_title("(c) Normalized opinion interface")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)


fig.suptitle("Echo silo diagnostics — time series")

plt.tight_layout()
plt.savefig(
    f"{OUT_DIR}/segregation_timeseries.png",
    dpi=180
)
plt.close()


# ============================================================
# PLOT 6: Angular segregation
# ============================================================
print("6/6: angular segregation...")
n_bins_ang = 36
bnd_r = np.zeros(n_t, dtype=int); bnd_l = np.zeros(n_t, dtype=int)
ang_r = np.full((n_t, n_bins_ang), np.nan); ang_l = np.full((n_t, n_bins_ang), np.nan)
for it in range(n_t):
    s = sign_mat[it]
    for side_m, prof, bnd in [(side0=="right", ang_r, bnd_r), (side0=="left", ang_l, bnd_l)]:
        sel = side_m & (s != 0)
        if sel.sum() >= 2:
            b, bins = angular_boundaries(theta_mat[it][sel], s[sel], n_bins=n_bins_ang)
            bnd[it] = b; prof[it] = bins

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
cmap_ang = plt.cm.RdYlBu_r.copy(); cmap_ang.set_bad("lightgrey")
for row, (prof, bnds, name) in enumerate([(ang_r, bnd_r, f"right (R={R_right})"), (ang_l, bnd_l, f"left (R={R_left})")]):
    ax = axes[row, 0]
    im = ax.imshow(prof.T, aspect="auto", origin="lower", extent=[times[0], times[-1], -np.pi, np.pi],
                   cmap=cmap_ang, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_ylabel("θ"); ax.set_title(f"{name} — angular opinion")
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi]); ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    plt.colorbar(im, ax=ax, label="majority sign")
    ax = axes[row, 1]
    ax.plot(times, bnds, color="darkorange", lw=2)
    ax.set_ylabel("# transitions"); ax.set_title(f"{name} — boundaries"); ax.grid(alpha=0.3)
fig.suptitle("Angular segregation per ring")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/angular_segregation.png", dpi=180); plt.close()


print(f"\nDone. All 5 plots written to {OUT_DIR}/")