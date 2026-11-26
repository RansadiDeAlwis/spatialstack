"""Generate the README figures from the per-sample eval logs.

Usage (from the project root):
    PYTHONPATH=<dir with matplotlib> env/bin/python scripts_repro/make_figures.py
Writes PNGs to docs/figures/.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# palette (validated: blue/orange pass CVD + normal-vision checks; orange < 3:1, so every bar is value-labelled)
BLUE, ORANGE, GRAY = "#257de0", "#f47230", "#b4b4b4"
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": TEXT, "figure.dpi": 150,
})

CATS = [("object_counting", "Obj. count"), ("object_abs_distance", "Abs. dist"), ("object_size_estimation", "Obj. size"),
        ("room_size_estimation", "Room size"), ("object_rel_distance", "Rel. dist"), ("object_rel_direction", "Rel. dir"),
        ("route_planning", "Route plan"), ("obj_appearance_order", "Appr. order")]
PAPER_VSI = [71.0, 55.6, 69.1, 68.2, 67.3, 84.1, 41.2, 83.5]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def vsi_df(run):
    f = sorted(glob.glob(os.path.join(ROOT, "results", run, "**", "*_samples_vsibench.jsonl"), recursive=True))[-1]
    rows = []
    for line in open(f):
        s = json.loads(line)["vsibench_score"]
        sc = s["accuracy"] if s.get("accuracy") is not None else s.get("MRA:.5:.95:.05")
        rows.append(dict(src=s["dataset"], qt=s["question_type"], score=sc))
    df = pd.DataFrame(rows)
    df["cat"] = df.qt.str.replace(r"_(easy|medium|hard)$", "", regex=True)
    return df


def vsi_scores(df):
    """Paper convention: RelDir = mean of easy/medium/hard; Avg = mean of the categories present."""
    out = {}
    for key, _ in CATS:
        g = df[df.cat == key]
        if len(g) == 0:
            continue
        out[key] = 100 * (g.groupby("qt").score.mean().mean() if key == "object_rel_direction" else g.score.mean())
    return out, sum(out.values()) / len(out)


def label_bars(ax, bars, fmt="{:.1f}", dy=0.6, size=9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=size, color=TEXT)


def grouped(ax, labels, a, b, la, lb, ca=GRAY, cb=BLUE, ylim=None, fmt="{:.1f}"):
    x = np.arange(len(labels)); w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, a, w, color=ca, label=la, zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, b, w, color=cb, label=lb, zorder=3)
    label_bars(ax, b1, fmt); label_bars(ax, b2, fmt)
    ax.set_xticks(x, labels); ax.yaxis.grid(True, color=GRID, zorder=0)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="upper left", ncol=2)


# ---------------------------------------------------------------- 1. architecture diagram
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, size=10.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=TEXT,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color=TEXT2, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=1.4))

    box(0.2, 2.6, 1.7, 1.0, "32 video frames\n+ question", "#f3f3f3", "#999999")
    # vision path
    box(2.6, 4.3, 2.2, 0.9, "Qwen3.5 ViT\n(patch 16, 2×2 merge)", "#dbe9fb", BLUE)
    # geometry path
    box(2.6, 0.5, 2.2, 2.9, "", "#efe7fb", "#6a4ce2")
    ax.text(3.7, 3.15, "VGGT-1B (frozen)", ha="center", fontsize=10.5, fontweight="bold", color=TEXT)
    for i, (l, y) in enumerate([(11, 2.35), (17, 1.55), (23, 0.75)]):
        box(2.85, y, 1.7, 0.55, f"layer {l}", "#ffffff", "#6a4ce2", size=10)
    arrow(1.9, 3.3, 2.6, 4.7); arrow(1.9, 2.9, 2.6, 2.2)
    ax.text(0.25, 2.2, "VGGT copy resized\nto 14 px patches", fontsize=8.5, color=TEXT2, va="top")

    # fusion MLPs
    for i, y in enumerate([2.35, 1.55, 0.75]):
        box(5.4, y, 1.75, 0.55, "fusion MLP", "#fdebd9", ORANGE, size=10)
        arrow(4.55, y + 0.27, 5.4, y + 0.27, color="#6a4ce2")
    ax.text(6.27, 3.0, "8192 → 4096 → 2560\n(zero-init output)", ha="center", fontsize=8.5, color=TEXT2)

    # LLM stack
    ys = [4.2, 3.4, 2.6, 1.8]
    names = ["LLM decoder layer 0", "LLM decoder layer 1", "LLM decoder layer 2", "LLM decoder layers 3 … 31"]
    for y, n in zip(ys, names):
        box(8.3, y - 0.3, 2.6, 0.6, n, "#e3f4e6", "#008a17", size=10)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(9.6, y0 - 0.3, 9.6, y1 + 0.3)
    arrow(4.8, 4.75, 9.6, 4.75); arrow(9.6, 4.75, 9.6, 4.5)
    ax.text(6.9, 4.85, "vision tokens (2560-d)", ha="center", fontsize=9, color=TEXT2)
    # injection arrows: added to output of layers 0,1,2 (between layers)
    for (yf, ytarget) in zip([2.62, 1.82, 1.02], [3.8, 3.0, 2.2]):
        ax.annotate("", xy=(8.3, ytarget), xytext=(7.15, yf),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, connectionstyle="arc3,rad=-0.15"))
        ax.text(8.12, ytarget + 0.02, "+", fontsize=14, color=ORANGE, fontweight="bold", ha="center", va="center")
    box(8.3, 0.5, 2.6, 0.6, "answer (greedy)", "#f3f3f3", "#999999", size=10)
    arrow(9.6, 1.5, 9.6, 1.1)
    ax.text(11.0, 3.0, "added only to\nimage-token\npositions,\nat prefill", fontsize=8.5, color=TEXT2, va="center")
    ax.text(6, 5.85, "SpatialStack: VGGT layers 11 / 17 / 23 are added into LLM layers 0 / 1 / 2",
            ha="center", fontsize=13, fontweight="bold", color=TEXT)
    save(fig, "architecture.png")


# ---------------------------------------------------------------- 2. reproduction overview
def fig_repro_overview():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [4, 2]})
    grouped(axes[0], ["VSI-Bench", "CV-Bench", "BLINK-Spatial", "SPAR-Bench"],
            [67.52, 85.53, 52.12, 71.39], [67.54, 85.69, 51.30, 68.40], "Paper", "Reproduced", ylim=(0, 100))
    axes[0].set_title("SpatialStack-5B (Qwen3.5)", loc="left"); axes[0].set_ylabel("score (%)")
    grouped(axes[1], ["VSI-Bench", "CV-Bench"], [60.9, 81.2], [60.62, 79.12], "Paper", "Reproduced", ylim=(0, 100))
    axes[1].set_title("SpatialStack-4B (Qwen2.5)", loc="left")
    save(fig, "reproduction_overview.png")


# ---------------------------------------------------------------- 3. VSI per category
def fig_vsi_categories(base_scores):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ours = [base_scores[k] for k, _ in CATS]
    grouped(ax, [n for _, n in CATS], PAPER_VSI, ours, "Paper (67.5 avg)", "Reproduced (67.54 avg)", ylim=(0, 100))
    ax.set_title("VSI-Bench per category, SpatialStack-5B: paper vs reproduced", loc="left"); ax.set_ylabel("score (%)")
    save(fig, "vsi_categories.png")


# ---------------------------------------------------------------- 4. scene source baseline vs LoRA
def fig_scene_source(base, lora):
    srcs = [("arkitscenes", "ARKitScenes"), ("scannet", "ScanNet"), ("scannetpp", "ScanNet++")]
    b = [vsi_scores(base[base.src == s])[1] for s, _ in srcs]
    l = [vsi_scores(lora[lora.src == s])[1] for s, _ in srcs]
    n = [int((base.src == s).sum()) for s, _ in srcs]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    grouped(ax, [f"{name}\n({k:,} Qs)" for (_, name), k in zip(srcs, n)], b, l, "Released model (baseline)",
            "+ ScanNet++ LoRA (ours)", ylim=(0, 88), fmt="{:.2f}", cb=ORANGE)
    for i, (x0, x1) in enumerate(zip(b, l)):
        ax.text(i, max(x0, x1) + 6, f"Δ {x1 - x0:+.2f}", ha="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_title("VSI-Bench average by scene source", loc="left"); ax.set_ylabel("score (%)")
    save(fig, "scene_source.png")
    return dict(zip([s for s, _ in srcs], zip(b, l, n)))


# ---------------------------------------------------------------- 5. LoRA delta per category
def fig_lora_delta(base, lora):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (title, sel) in zip(axes, [("ScanNet++ scenes (1,458 Qs, trained domain)", "scannetpp"),
                                         ("Full VSI-Bench (5,130 Qs)", None)]):
        bb = base if sel is None else base[base.src == sel]
        ll = lora if sel is None else lora[lora.src == sel]
        sb, ab = vsi_scores(bb); sl, al = vsi_scores(ll)
        labels = [n for k, n in CATS] + ["Average"]
        deltas = [sl[k] - sb[k] for k, _ in CATS] + [al - ab]
        y = np.arange(len(labels))[::-1]
        colors = [BLUE if d >= 0 else ORANGE for d in deltas]
        ax.barh(y, deltas, color=colors, height=0.62, zorder=3)
        for yi, d in zip(y, deltas):
            ax.text(d + (0.25 if d >= 0 else -0.25), yi, f"{d:+.2f}", va="center", ha="left" if d >= 0 else "right",
                    fontsize=9.5, color=TEXT, fontweight="bold" if yi == 0 else "normal")
        ax.axvline(0, color=TEXT2, lw=1); ax.xaxis.grid(True, color=GRID, zorder=0)
        ax.set_yticks(y, labels); ax.set_title(title, loc="left"); ax.set_xlabel("Δ score vs released model (points)")
        ax.set_xlim(-14, 14)  # same scale in both panels
    fig.suptitle("Effect of the ScanNet++ LoRA fine-tune per VSI-Bench category", x=0.01, ha="left",
                 fontweight="bold", fontsize=13, color=TEXT)
    save(fig, "lora_delta.png")


# ---------------------------------------------------------------- 6. training loss
def fig_loss():
    p = "/mnt/4dgs/spatialstack/lora_runs/scannetpp_lora5k/lora_stats.jsonl"
    st = pd.DataFrame([json.loads(l) for l in open(p)])
    st = st[st.loss.notna()]
    ma = st.loss.rolling(20, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.step, st.loss, color=GRAY, lw=1, label="per step", zorder=2)
    ax.plot(st.step, ma, color=BLUE, lw=2.2, label="20-step moving average", zorder=3)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_xlabel("optimizer step (batch 16)"); ax.set_ylabel("training loss")
    ax.set_title(f"LoRA training loss: 5,000 ScanNet++ QAs, 1 epoch, {int(st.step.max())} steps", loc="left")
    ax.legend(frameon=False)
    save(fig, "training_loss.png")
    return st


# ---------------------------------------------------------------- 7. training example
def fig_example():
    import decord
    vid = os.path.join(ROOT, "repo/data/vlm3r/media/ScanNetpp/videos/train/da1df716a5.mp4")
    vr = decord.VideoReader(vid)
    idx = np.linspace(0, len(vr) - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2))
    for ax, i, k in zip(axes.flat, idx, range(8)):
        ax.imshow(vr[int(i)].asnumpy()); ax.axis("off"); ax.set_title(f"frame {k + 1}", fontsize=9, color=TEXT2)
    fig.suptitle("Q: What is the direct distance between the mouse and the pillow (in meters)?     A: 3.3",
                 fontsize=12.5, fontweight="bold", color=TEXT, y=0.99)
    fig.text(0.5, 0.005, "One of the 5,000 ScanNet++ training QAs: the 8 uniformly sampled frames the model sees; "
             "the answer is measured from the room's 3D scan.", ha="center", fontsize=9.5, color=TEXT2)
    plt.subplots_adjust(wspace=0.03, hspace=0.12)
    save(fig, "training_example.png")


if __name__ == "__main__":
    base, lora = vsi_df("spatialstack"), vsi_df("lora")
    bscores, bavg = vsi_scores(base)
    fig_architecture(); fig_repro_overview(); fig_vsi_categories(bscores)
    src = fig_scene_source(base, lora); fig_lora_delta(base, lora); fig_loss()
    try:
        fig_example()
    except Exception as e:  # video not available on this machine
        print("skipped training_example:", e, file=sys.stderr)
    print("baseline avg", round(bavg, 2), "lora avg", round(vsi_scores(lora)[1], 2), src)
