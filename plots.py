
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
from matplotlib.patches import Circle


# ============================================================
# PARAMETERS
# ============================================================

R_right = 0.5
R_left = 0.5

K_NN = 8
LINK_THR = 0.05

N_BINS_ANG = 36
MAX_EMPTY_BINS = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def knn_segregation(pos, signs, k=K_NN):

    n = len(pos)

    if n < 2:
        return (
            np.full(n, np.nan),
            np.full(n, np.nan)
        )

    k = min(k, n - 1)

    _, idx = cKDTree(pos).query(pos, k=k + 1)

    mean_opinion = np.full(n, np.nan)
    same_fraction = np.full(n, np.nan)

    for i in range(n):

        # Explicitly exclude the individual itself
        neighbours = idx[i]
        neighbours = neighbours[neighbours != i][:k]

        if len(neighbours) == 0:
            continue

        nn_sig = signs[neighbours]

        mean_opinion[i] = nn_sig.mean()

        same_fraction[i] = (
            nn_sig == signs[i]
        ).mean()

    return mean_opinion, same_fraction


def angular_boundaries(
    theta,
    opinions,
    n_bins=36,
    max_empty_bins=1
):

    if len(theta) < 2:
        return np.nan, np.full(n_bins, np.nan)

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    bin_idx = np.digitize(theta, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Store mean opinion strength
    bin_opinion = np.full(n_bins, np.nan)

    for ib in range(n_bins):

        sel = bin_idx == ib

        if not sel.any():
            continue

        bin_opinion[ib] = np.mean(opinions[sel])

    # Determine majority sign for boundary calculation
    bin_sign = np.sign(bin_opinion)

    # Empty and tied bins
    bin_sign[np.isnan(bin_opinion)] = np.nan
    bin_sign[bin_opinion == 0] = np.nan

    occupied = np.where(~np.isnan(bin_sign))[0]

    if len(occupied) < 2:
        return np.nan, bin_opinion

    trans = 0

    for i in range(len(occupied)):

        a_idx = occupied[i]
        b_idx = occupied[(i + 1) % len(occupied)]

        gap = (b_idx - a_idx - 1) % n_bins

        if gap <= max_empty_bins:

            if bin_sign[a_idx] != bin_sign[b_idx]:
                trans += 1

    return trans, bin_opinion


# ============================================================
# PLOT ONE EXPERIMENT
# ============================================================

def plot_experiment(run_name):

    RUN_DIR = os.path.join("results", run_name)
    OUT_DIR = os.path.join(RUN_DIR, "plots")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nPlotting: {run_name}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        os.path.join(RUN_DIR, "individual_opinions.csv")
    )

    df["sign"] = np.sign(df["infection"]).astype(int)
    df["is_infected"] = df["sign"] != 0

    times = np.sort(df["time"].unique())

    numparts = df["agent_num"].nunique()

    if len(times) == 0:
        print("No simulation data found.")
        return

    groups = {
        t: d for t, d in df.groupby("time", sort=True)
    }

    sign_mat = df.pivot(
        index="time",
        columns="agent_num",
        values="sign"
    ).to_numpy()

    x_mat = df.pivot(
        index="time",
        columns="agent_num",
        values="x"
    ).to_numpy()

    y_mat = df.pivot(
        index="time",
        columns="agent_num",
        values="y"
    ).to_numpy()

    n_t, n_a = sign_mat.shape

    df_opinions = df.pivot(
    index="time",
    columns="agent_num",
    values="infection"
        ).to_numpy()

    # Determine ring from the current x-coordinate.
    # This follows the same left/right selection used in u_2_annuli().
    #
    # x >= 0: right ring
    # x < 0: left ring

    right_mask = x_mat >= 0

    cx_mat = np.where(
        right_mask,
        R_right,
        -R_left
    )

    theta_mat = np.arctan2(
        y_mat,
        x_mat - cx_mat
    )

    pos_count = (sign_mat == 1).sum(axis=1)
    neg_count = (sign_mat == -1).sum(axis=1)

    # --------------------------------------------------------
    # 1. SPATIAL SNAPSHOTS
    # --------------------------------------------------------

    print("1/5: Spatial snapshots...")

    snap_idx = [
        0,
        len(times)//4,
        len(times)//2,
        3*len(times)//4,
        len(times)-1
    ]

    fig, axes = plt.subplots(1, 5, figsize=(19, 4))

    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)

    for ax, t in zip(axes, times[snap_idx]):

        d = groups[t]

        und = d[d["sign"] == 0]
        com = d[d["sign"] != 0]

        ax.scatter(
            und["x"],
            und["y"],
            color="black",
            s=3,
            alpha=0.4
        )

        ax.scatter(
            com["x"],
            com["y"],
            c=com["infection"],
            cmap=cmap,
            norm=norm,
            s=15
        )

        ax.add_patch(
            Circle(
                (R_right, 0),
                R_right,
                fill=False,
                edgecolor="grey",
                linestyle="--",
                linewidth=0.6,
                alpha=0.6
            )
        )

        ax.add_patch(
            Circle(
                (-R_left, 0),
                R_left,
                fill=False,
                edgecolor="grey",
                linestyle="--",
                linewidth=0.6,
                alpha=0.6
            )
        )

        ax.set_title(f"t = {t:.2f}")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array([])

    fig.subplots_adjust(
        right=0.88,
        wspace=0.25,
        top=0.82,
        bottom=0.15
    )

    cbar_ax = fig.add_axes([0.91, 0.18, 0.015, 0.60])

    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        ticks=[-1, -0.5, 0, 0.5, 1]
    )

    cbar.set_label("Opinion strength")

    fig.suptitle(f"Spatial Spread of Opinions - {run_name}")

    plt.savefig(
        f"{OUT_DIR}/spatial_snapshots.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 2. DISTRIBUTION OF OPINIONS
    # --------------------------------------------------------

    print("2/5: Opinion distributions...")

    snap_idx = [
        0,
        len(times)//4,
        len(times)//2,
        (len(times)*3)//4,
        len(times)-1
    ]

    fig, axes = plt.subplots(1, 5, figsize=(12, 6))

    bin_edges = np.linspace(-1, 1, 41)

    for ax, t in zip(axes.flatten(), times[snap_idx]):

        com = groups[t].loc[
            groups[t]["sign"] != 0,
            "infection"
        ].to_numpy()

        ax.hist(
            com,
            bins=bin_edges,
            color="steelblue",
            edgecolor="black",
            alpha=0.8
        )

        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("Opinion value")
        ax.set_ylabel("Count")
        ax.set_xlim(-1, 1)

    fig.suptitle(f"Distribution of Opinion Strength - {run_name}")

    plt.tight_layout()

    plt.savefig(
        f"{OUT_DIR}/distribution.png",
        dpi=200
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 3. PHASE PORTRAIT
    # --------------------------------------------------------

    print("3/5: Phase portrait...")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        pos_count,
        neg_count,
        color="steelblue",
        linewidth=2
    )

    ax.scatter(
        [pos_count[0]],
        [neg_count[0]],
        color="green",
        s=70,
        zorder=5,
        label="start"
    )

    ax.scatter(
        [pos_count[-1]],
        [neg_count[-1]],
        color="red",
        s=70,
        zorder=5,
        label="end"
    )

    ax.set_xlabel("N+")
    ax.set_ylabel("N-")
    ax.set_title("Phase portrait: (N+, N-)")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUT_DIR}/phase_portrait.png",
        dpi=200
    )

    plt.close(fig)

    # --------------------------------------------------------
    # ECHO SILO DIAGNOSTICS
    # --------------------------------------------------------

    print("Computing echo-silo diagnostics...")

    same_frac_pos = np.full(n_t, np.nan)
    same_frac_neg = np.full(n_t, np.nan)

    excess_pos = np.full(n_t, np.nan)
    excess_neg = np.full(n_t, np.nan)

    interface_count = np.zeros(n_t, dtype=int)
    same_close_count = np.zeros(n_t, dtype=int)

    for it in range(n_t):

        s = sign_mat[it]

        xy = np.column_stack([
            x_mat[it],
            y_mat[it]
        ])

        comm = s != 0

        n_pos = int((s == 1).sum())
        n_neg = int((s == -1).sum())

        n_c = n_pos + n_neg

        if n_c < 2:
            continue

        pos_c = xy[comm]
        sign_c = s[comm]

        _, same_frac = knn_segregation(
            pos_c,
            sign_c,
            K_NN
        )

        glob_pos = (n_pos - 1) / (n_c - 1)
        glob_neg = (n_neg - 1) / (n_c - 1)

        if n_pos > 0:

            m = sign_c == 1

            same_frac_pos[it] = same_frac[m].mean()

            excess_pos[it] = (
                same_frac[m].mean() - glob_pos
            )

        if n_neg > 0:

            m = sign_c == -1

            same_frac_neg[it] = same_frac[m].mean()

            excess_neg[it] = (
                same_frac[m].mean() - glob_neg
            )

        pairs = cKDTree(pos_c).query_pairs(
            r=LINK_THR
        )

        if pairs:

            i_arr, j_arr = map(
                np.array,
                zip(*pairs)
            )

            same = int(
                (sign_c[i_arr] == sign_c[j_arr]).sum()
            )

            same_close_count[it] = same

            interface_count[it] = len(pairs) - same

    # --------------------------------------------------------
    # 4. SEGREGATION TIME SERIES
    # --------------------------------------------------------

    print("4/5: Segregation time series...")

    n_infected = (sign_mat != 0).sum(axis=1)
    n_positive = (sign_mat == 1).sum(axis=1)

    glob_pos_arr = np.full(n_t, np.nan)

    valid = n_infected >= 2

    glob_pos_arr[valid] = (
        (n_positive[valid] - 1)
        /
        (n_infected[valid] - 1)
    )

    cross_pair_fraction = np.full(n_t, np.nan)

    for it in range(n_t):

        total_pairs = (
            interface_count[it]
            +
            same_close_count[it]
        )

        if total_pairs > 0:

            cross_pair_fraction[it] = (
                interface_count[it] / total_pairs
            )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) k-NN segregation
    ax = axes[0]

    ax.plot(
        times,
        same_frac_pos,
        color="red",
        label="+1 same-sign NN"
    )

    ax.plot(
        times,
        same_frac_neg,
        color="blue",
        label="-1 same-sign NN"
    )

    ax.plot(
        times,
        glob_pos_arr,
        color="red",
        linestyle=":",
        alpha=0.5,
        label="random +1"
    )

    ax.plot(
        times,
        1 - glob_pos_arr,
        color="blue",
        linestyle=":",
        alpha=0.5,
        label="random -1"
    )

    ax.set_ylabel("Same-sign NN fraction")
    ax.set_xlabel("Time")
    ax.set_title("(a) k-NN segregation")
    ax.set_ylim(0, 1)

    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Segregation excess
    ax = axes[1]

    ax.plot(
        times,
        excess_pos,
        color="red",
        lw=2,
        label="+1 excess"
    )

    ax.plot(
        times,
        excess_neg,
        color="blue",
        lw=2,
        label="-1 excess"
    )

    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.axhline(0.2, color="orange", ls=":", lw=0.8)
    ax.axhline(-0.2, color="orange", ls=":", lw=0.8)

    ax.set_ylabel("Observed - random")
    ax.set_xlabel("Time")
    ax.set_title("(b) Segregation excess")

    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Normalized cross-opinion pairs
    ax = axes[2]

    ax.plot(
        times,
        cross_pair_fraction,
        color="darkorange",
        lw=2,
        label="Cross-opinion pair fraction"
    )

    ax.axhline(
        0.5,
        color="grey",
        linestyle="--",
        lw=0.8,
        label="50% reference"
    )

    ax.set_ylabel("Fraction of cross-opinion pairs")
    ax.set_xlabel("Time")
    ax.set_title("(c) Normalized opinion interface")
    ax.set_ylim(0, 1)

    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Echo silo diagnostics - {run_name}")

    plt.tight_layout()

    plt.savefig(
        f"{OUT_DIR}/segregation_timeseries.png",
        dpi=180
    )

    plt.close(fig)


    # --------------------------------------------------------
    # 5. ANGULAR SEGREGATION
    # --------------------------------------------------------

    print("5/5: Angular segregation...")

    n_bins_ang = N_BINS_ANG

    # Initialize with NaN because insufficient data
    # should not be interpreted as zero boundaries.

    bnd_r = np.full(n_t, np.nan)
    bnd_l = np.full(n_t, np.nan)

    ang_r = np.full(
        (n_t, n_bins_ang),
        np.nan
    )

    ang_l = np.full(
        (n_t, n_bins_ang),
        np.nan
    )

    for it in range(n_t):

        s = sign_mat[it]

        # Determine ring using current x-coordinate
        right_mask = x_mat[it] >= 0
        left_mask = x_mat[it] < 0

        for side_m, prof, bnd in [
            (right_mask, ang_r, bnd_r),
            (left_mask, ang_l, bnd_l)
        ]:

            # Only consider opinionated individuals
            sel = side_m & (s != 0)

            # Require at least two opinionated individuals
            if sel.sum() < 2:
                continue

            # Calculate mean opinion strength in each angular bin.
            # Boundary count is calculated using majority sign.
            b, bins = angular_boundaries(
                theta_mat[it][sel],
                df_opinions[it][sel],
                n_bins=n_bins_ang,
                max_empty_bins=MAX_EMPTY_BINS
            )

            bnd[it] = b
            prof[it] = bins


    # --------------------------------------------------------
    # PLOT ANGULAR SEGREGATION
    # --------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    cmap_ang = plt.cm.RdYlBu_r.copy()
    cmap_ang.set_bad("lightgrey")

    for row, (prof, bnds, name) in enumerate([
        (ang_r, bnd_r, f"right (R={R_right})"),
        (ang_l, bnd_l, f"left (R={R_left})")
    ]):

        # ----------------------------------------------------
        # Angular opinion distribution
        # ----------------------------------------------------

        ax = axes[row, 0]

        im = ax.imshow(
            prof.T,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                -np.pi,
                np.pi
            ],
            cmap=cmap_ang,
            vmin=-1,
            vmax=1,
            interpolation="nearest"
        )

        ax.set_ylabel("Theta")
        ax.set_xlabel("Time")

        ax.set_title(
            f"{name} - angular opinion strength"
        )

        ax.set_yticks([
            -np.pi,
            -np.pi/2,
            0,
            np.pi/2,
            np.pi
        ])

        ax.set_yticklabels([
            "-π",
            "-π/2",
            "0",
            "π/2",
            "π"
        ])

        plt.colorbar(
            im,
            ax=ax,
            label="Mean opinion strength"
        )

        # ----------------------------------------------------
        # Angular boundaries
        # ----------------------------------------------------

        ax = axes[row, 1]

        ax.plot(
            times,
            bnds,
            color="darkorange",
            lw=2
        )

        ax.set_ylabel("# transitions")
        ax.set_xlabel("Time")

        ax.set_title(
            f"{name} - angular boundaries"
        )

        ax.grid(alpha=0.3)


    fig.suptitle(
        f"Angular segregation per ring - {run_name}"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUT_DIR}/angular_segregation.png",
        dpi=180
    )

    plt.close(fig)



# ============================================================
# AUTOMATICALLY PLOT ALL EXPERIMENTS
# ============================================================

if __name__ == "__main__":

    results_dir = "results"

    if not os.path.isdir(results_dir):
        raise FileNotFoundError("Could not find results directory.")

    csv_file = os.path.join(
        results_dir,
        "individual_opinions.csv"
    )

    if not os.path.isfile(csv_file):
        raise FileNotFoundError(
            f"Could not find {csv_file}"
        )

    print("Found simulation data.")

    plot_experiment("")

    print("\nPlotting complete!")