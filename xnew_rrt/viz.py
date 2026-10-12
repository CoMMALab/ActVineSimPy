'''
Headless drawing for the planner and the rollout viewer. Everything is written to image / gif /
html files (Agg backend), so it works over SSH: open the files in the VS Code editor.

Plots are in millimeters (= dvsim internal length units).
'''

import math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
from PIL import Image

from dvsim import si

MM = si.len_to_mm  # internal length -> mm

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
WALL_FACE, WALL_EDGE = "#e1e0d9", "#898781"
VINE = "#2a78d6"       # the vine being shown
BOX = "#eb6834"        # movable boxes
GOAL = "#1baf7a"
BEST = "#6250d6"       # tip trace of the highlighted branch
# tree depth: one hue, light -> dark
DEPTH_CMAP = LinearSegmentedColormap.from_list("depth", ["#86b6ef", "#2a78d6", "#0d366b"])


def _box_corners_mm(pose, hw, hh):
    cx, cy, th = pose
    c, s = math.cos(th), math.sin(th)
    return [(MM(cx + c * lx - s * ly), MM(cy + s * lx + c * ly))
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def new_figure(scene, max_dim_in=11):
    xmin, xmax, ymin, ymax = [v * 1000 for v in scene.bounds_m]
    aspect = (ymax - ymin) / (xmax - xmin)
    size = (max_dim_in, max_dim_in * aspect) if aspect <= 1 else (max_dim_in / aspect, max_dim_in)
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", color=INK_2); ax.set_ylabel("y (mm)", color=INK_2)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    return fig, ax


def draw_scene(ax, sim, objects_state=None):
    '''Static walls, goal, start marker and (optionally) the movable boxes of one state.'''
    scene = sim.scene
    for (pose, hw, hh) in sim.static_boxes():
        ax.add_patch(Polygon(_box_corners_mm(pose, hw, hh), closed=True, facecolor=WALL_FACE,
                             edgecolor=WALL_EDGE, lw=1, zorder=1))

    gx, gy = scene.goal_m[0] * 1000, scene.goal_m[1] * 1000
    ax.add_patch(Circle((gx, gy), scene.goal_radius_m * 1000, facecolor=GOAL, alpha=0.15, zorder=2))
    ax.add_patch(Circle((gx, gy), scene.goal_radius_m * 1000, fill=False, edgecolor=GOAL, lw=2,
                        ls="--", zorder=2))
    ax.annotate("goal", (gx, gy), ha="center", va="center", color=INK_2, fontsize=9, zorder=2)

    sx, sy = scene.start[0] * 1000, scene.start[1] * 1000
    ax.plot([sx], [sy], marker="^", ms=9, color=INK, zorder=9, ls="none")
    ax.annotate("start", (sx, sy), xytext=(0, -14), textcoords="offset points", ha="center",
                color=INK_2, fontsize=9, zorder=9)

    if scene.box_goal_m is not None:
        bx, by, br = [v * 1000 for v in scene.box_goal_m]
        ax.add_patch(Circle((bx, by), br, fill=False, edgecolor=BOX, lw=2, ls="--", zorder=2))
        ax.annotate("box goal", (bx, by), ha="center", va="center", color=INK_2, fontsize=9, zorder=2)

    if objects_state is not None:
        draw_objects(ax, sim, objects_state)


def draw_objects(ax, sim, state, alpha=0.85, outline_only=False, zorder=4):
    for (pose, hw, hh) in sim.object_boxes(state):
        ax.add_patch(Polygon(_box_corners_mm(pose, hw, hh), closed=True,
                             facecolor="none" if outline_only else BOX,
                             edgecolor=BOX if outline_only else SURFACE,
                             lw=1 if outline_only else 2, alpha=alpha, zorder=zorder))


def vine_xy_mm(state):
    n = state.num_bodies
    xs = [MM(float(state.state[0, 3 * j])) for j in range(n)]
    ys = [MM(float(state.state[0, 3 * j + 1])) for j in range(n)]
    return xs, ys


def draw_vine(ax, sim, state, color=VINE, alpha=0.35, zorder=5):
    xs, ys = vine_xy_mm(state)
    radius_mm = sim.cfg.vine_radius_m * 1000
    ax.add_collection(PatchCollection([Circle((x, y), radius_mm) for x, y in zip(xs, ys)],
                                      facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))
    base = (MM(float(sim.base_of(state)[0])), MM(float(sim.base_of(state)[1])))
    ax.plot([base[0]] + xs, [base[1]] + ys, "-", color=color, lw=2, zorder=zorder + 1)
    ax.add_patch(Circle((xs[-1], ys[-1]), radius_mm, fill=False, edgecolor=color, lw=2, zorder=zorder + 1))


def _tip_mm(state):
    x, y, _ = state.tip()
    return MM(float(x)), MM(float(y))


def edge_trace_mm(node):
    '''Tip positions along the edge parent -> node (starting at the parent's tip).'''
    pts = [_tip_mm(node.parent.info)] if node.parent is not None else []
    return pts + [_tip_mm(s) for s in node.edge]


def draw_tree(ax, sim, tree, highlight=None, sample_mm=None, new_node=None, title=None):
    '''
    The whole RRT in the workspace:
      - every edge as the path the vine TIP actually took while that edge was simulated,
        shaded by depth in the tree (light = near the root)
      - where each leaf left the movable boxes (thin outlines)
      - `highlight` (a Node): its branch from the root, the vine shape and the boxes at that node
      - the last random sample (x) and the node it produced (ring)
    '''
    draw_scene(ax, sim)

    nodes = [n for n in tree.all_nodes if n.parent is not None]
    max_depth = max([n.depth for n in nodes], default=1)
    norm = Normalize(vmin=0, vmax=max(max_depth, 1))

    if nodes:
        ax.add_collection(LineCollection([edge_trace_mm(n) for n in nodes],
                                         colors=[DEPTH_CMAP(norm(n.depth)) for n in nodes],
                                         linewidths=1.2, zorder=6))
        ends = [_tip_mm(n.info) for n in nodes]
        ax.scatter([p[0] for p in ends], [p[1] for p in ends], s=9, zorder=7,
                   c=[DEPTH_CMAP(norm(n.depth)) for n in nodes], edgecolors=SURFACE, linewidths=0.4)

    for leaf in tree.all_leaves:
        draw_objects(ax, sim, leaf.info, alpha=0.25, outline_only=True, zorder=3)

    if highlight is not None:
        draw_objects(ax, sim, highlight.info)
        draw_vine(ax, sim, highlight.info, alpha=0.22, zorder=4)
        trace = []
        for n in highlight.path_from_root():
            trace += edge_trace_mm(n)
        if trace:
            ax.plot([p[0] for p in trace], [p[1] for p in trace], "-", color=BEST, lw=2.5, zorder=8)
    else:
        draw_objects(ax, sim, tree.root.info)

    if sample_mm is not None:
        ax.plot([sample_mm[0]], [sample_mm[1]], marker="x", ms=9, mew=2, color=INK, ls="none", zorder=10)
        # samples that also contain box positions: (tip x, tip y, box0 x, box0 y, ...)
        ax.plot(sample_mm[2::2], sample_mm[3::2], marker="x", ms=9, mew=2, color=BOX, ls="none", zorder=10)
    if new_node is not None:
        x, y = _tip_mm(new_node.info)
        ax.plot([x], [y], marker="o", ms=11, mfc="none", mec=INK, mew=1.5, ls="none", zorder=10)

    handles = [Line2D([], [], color=DEPTH_CMAP(0.6), lw=1.2, marker="o", ms=3, label="tree: tip path per edge"),
               Line2D([], [], color=BEST, lw=2.5, label="branch closest to goal"),
               Patch(facecolor=VINE, alpha=0.3, label="vine at that node"),
               Patch(facecolor=BOX, label="movable box at that node"),
               Patch(facecolor="none", edgecolor=BOX, alpha=0.5, label="box at other leaves"),
               Line2D([], [], color=INK, marker="x", ms=7, mew=2, ls="none", label="last sample"),
               Line2D([], [], color=INK, marker="o", ms=8, mfc="none", ls="none", label="node it added")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=8, labelcolor=INK_2)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=DEPTH_CMAP); sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.45, pad=0.02, anchor=(0.0, 0.0))
    cbar.set_label("depth in tree (edges from root)", color=INK_2, fontsize=8)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_visible(False)

    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def draw_rollout_frame(ax, sim, state, tip_trace_mm=None, title=None):
    '''One frame of a single rollout: scene, boxes, vine, and the tip's path so far.'''
    draw_scene(ax, sim, objects_state=state)
    if tip_trace_mm and len(tip_trace_mm) > 1:
        ax.plot([p[0] for p in tip_trace_mm], [p[1] for p in tip_trace_mm], "-", color=BEST, lw=1.5, zorder=8)
    draw_vine(ax, sim, state)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def save_figure(fig, path, dpi=80):
    '''Write atomically so an image viewer that is watching the file never reads half a png.'''
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor())
    os.replace(tmp, path)
    plt.close(fig)


def write_gif(frame_paths, out_path, frame_ms=80, hold_last_ms=1500):
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    w, h = imgs[0].size
    imgs = [im if im.size == (w, h) else im.resize((w, h)) for im in imgs]
    durations = [frame_ms] * (len(imgs) - 1) + [hold_last_ms]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=durations, loop=0)


def write_html_player(frame_paths, out_path, title="rollout", frame_ms=80):
    '''Self-contained html page with play/pause and a frame scrubber (frames are inlined).'''
    import base64
    uris = []
    for p in frame_paths:
        with open(p, "rb") as f:
            uris.append("data:image/png;base64," + base64.b64encode(f.read()).decode())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;background:{SURFACE};color:{INK};margin:16px}}
img{{max-width:100%;border:1px solid #e1e0d9}} input[type=range]{{width:60%}}</style></head><body>
<h3>{title}</h3><img id="f"><br>
<button id="p">pause</button> <input id="s" type="range" min="0" max="{len(uris) - 1}" value="0">
<span id="n"></span>
<script>
const F={uris!r};let i=0,play=true;
const img=document.getElementById('f'),s=document.getElementById('s'),n=document.getElementById('n'),b=document.getElementById('p');
function show(k){{i=k;img.src=F[i];s.value=i;n.textContent=(i+1)+' / '+F.length;}}
s.oninput=()=>{{play=false;b.textContent='play';show(+s.value);}};
b.onclick=()=>{{play=!play;b.textContent=play?'pause':'play';}};
document.onkeydown=e=>{{if(e.key==='ArrowRight'){{play=false;b.textContent='play';show(Math.min(i+1,F.length-1));}}
 if(e.key==='ArrowLeft'){{play=false;b.textContent='play';show(Math.max(i-1,0));}}}};
setInterval(()=>{{if(play)show((i+1)%F.length);}},{frame_ms});show(0);
</script></body></html>"""
    with open(out_path, "w") as f:
        f.write(html)
