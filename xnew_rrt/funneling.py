'''
Do vines funnel? Measured on finished trees (tree_log.pkl from run_envs.py).

Hypothesis: contacts make different controls end in the same place, so a tree fills up with
duplicate states, and a little logic to keep the tree sparse should go a long way. Two measures:
  duplicates   fraction of nodes whose (tip, box centers) lies within d mm of an earlier node's,
               for d = 1, 2, 5, 10 mm (the boxes count with the same weight as the tip)
  siblings     pairs of children of one node, grown with curvatures at least 1 1/m apart: the
               fraction whose tips end within 2 mm of each other
Both are split by whether the node's tip is in contact (its circle within 5 mm of a wall or box)
or free.
Run from the repo root:
    python xnew_rrt/funneling.py out/envs
Writes out/funneling.json and prints a table.
'''

import argparse, json, os, pickle

import numpy as np
from scipy.spatial import cKDTree

RADII_MM = (1.0, 2.0, 5.0, 10.0)


CONTACT_MM = 5.0


def _segment_dist(p, a, b):
    ab, ap = b - a, p - a
    t = np.clip((ap @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(ap - t * ab)


def poly_dist(p, poly):
    '''Distance from point p to a convex polygon (0 inside).'''
    poly = np.asarray(poly, dtype=float)
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    cross = [(b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) for a, b in edges]
    inside = all(c >= 0 for c in cross) or all(c <= 0 for c in cross)
    return 0.0 if inside else min(_segment_dist(p, a, b) for a, b in edges)


def in_contact(node, walls, radius):
    tip = np.asarray(node["tip_trace"][-1], dtype=float)
    return any(poly_dist(tip, poly) <= radius + CONTACT_MM for poly in list(walls) + list(node["boxes"]))


def analyze(log):
    nodes = [n for n in log["nodes"] if n["parent"] is not None and not n.get("reset") and len(n["tip_trace"])]
    feats = np.array([np.concatenate([n["tip_trace"][-1], n["box_centers"].reshape(-1)]) for n in nodes])
    walls, radius = log["scene"]["walls_mm"], log["scene"]["vine_radius_mm"]
    contact = np.array([in_contact(n, walls, radius) for n in nodes])
    out = {"nodes": len(nodes), "contact_frac": float(contact.mean())}
    # distance to the nearest EARLIER node (node order = creation order)
    k = min(32, len(feats))
    dist, idx = cKDTree(feats).query(feats, k=k)
    earlier = np.where(idx < np.arange(len(feats))[:, None], dist, np.inf).min(1)
    for i in np.flatnonzero(~np.isfinite(earlier))[1:]:          # no earlier node among the k nearest
        earlier[i] = np.linalg.norm(feats[:i] - feats[i], axis=1).min()
    out["duplicate_frac"] = {str(d): float((earlier[1:] <= d).mean()) for d in RADII_MM}
    for name, mask in (("contact", contact), ("free", ~contact)):
        m = mask[1:]
        out[f"duplicate_frac_{name}"] = {str(d): float((earlier[1:][m] <= d).mean()) if m.any() else None
                                         for d in RADII_MM}

    # sibling pairs
    by_parent = {}
    for n, f in zip(nodes, feats):
        by_parent.setdefault(n["parent"], []).append((n, f))
    pairs = collapsed = 0
    split = {"contact": [0, 0], "free": [0, 0]}
    index = {id(n): i for i, n in enumerate(nodes)}
    for kids in by_parent.values():
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                (na, fa), (nb, fb) = kids[a], kids[b]
                if abs(na["control"] - nb["control"]) < 1.0:
                    continue
                pairs += 1
                same = np.linalg.norm(fa[:2] - fb[:2]) <= 2.0
                collapsed += same
                kind = "contact" if contact[index[id(na)]] or contact[index[id(nb)]] else "free"
                split[kind][0] += 1
                split[kind][1] += same
    out["sibling_pairs"] = pairs
    out["sibling_collapsed_frac"] = collapsed / max(pairs, 1)
    for kind, (n_pairs, n_same) in split.items():
        out[f"sibling_pairs_{kind}"] = n_pairs
        out[f"sibling_collapsed_frac_{kind}"] = n_same / max(n_pairs, 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", default="out/envs")
    parser.add_argument("--out", default="out/funneling.json")
    args = parser.parse_args()
    rows = {}
    for name in sorted(os.listdir(args.run_dir)):
        path = os.path.join(args.run_dir, name, "tree_log.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            log = pickle.load(f)
        rows[name] = r = analyze(log)
        pct = lambda v: "  -  " if v is None else f"{100 * v:4.0f}%"
        print(f"{name:>3}  {r['nodes']:6d} nodes, {100 * r['contact_frac']:3.0f}% in contact | within 2 mm of an "
              f"earlier node: all {pct(r['duplicate_frac']['2.0'])}, contact {pct(r['duplicate_frac_contact']['2.0'])}, "
              f"free {pct(r['duplicate_frac_free']['2.0'])} | sibling pairs, same tip: contact "
              f"{pct(r['sibling_collapsed_frac_contact'])} of {r['sibling_pairs_contact']}, free "
              f"{pct(r['sibling_collapsed_frac_free'])} of {r['sibling_pairs_free']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
