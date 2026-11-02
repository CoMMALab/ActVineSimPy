'''
Layout of out/solutions/: one folder per kind of file, every file named <env>.<what>.<ext>, so all
envs' videos (or results, ...) can be looked at side by side:

    out/solutions/videos/   5a.solution.mp4 / .webm (VS Code plays the webm), 1c.solution.mp4, ...
                  gifs/     solution.gif and tree.gif per env
                  images/   env.png, plan.png, pushes.png, tree_final.png, ...
                  results/  <env>.result.json, summary.json
                  data/     tree_log.pkl (redraws the tree without planning again)
                  logs/

A run writes into a scratch folder (work_dir) and publish() moves its files into place, after
removing that env's files from the previous run.
'''

import os, shutil

KINDS = {".mp4": "videos", ".webm": "videos", ".gif": "gifs", ".png": "images", ".json": "results", ".pkl": "data",
         ".log": "logs", ".html": "html"}


def work_dir(root, name):
    path = os.path.join(root, ".work", name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def clear(root, name):
    '''Remove env `name`'s files (<name>.*) from every kind folder, except a console log (<name>.run.log)
    that the run itself may still be writing.'''
    for kind in set(KINDS.values()) | {"other"}:
        d = os.path.join(root, kind)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith(name + ".") and not f.endswith(".run.log"):
                    os.remove(os.path.join(d, f))


def publish(work, root, name):
    '''Move every file of a run's scratch folder to <root>/<kind>/<name>.<file>.'''
    clear(root, name)
    for f in sorted(os.listdir(work)):
        src = os.path.join(work, f)
        if not os.path.isfile(src):
            continue
        kind = KINDS.get(os.path.splitext(f)[1], "other")
        os.makedirs(os.path.join(root, kind), exist_ok=True)
        os.replace(src, os.path.join(root, kind, f"{name}.{f}"))
    shutil.rmtree(work, ignore_errors=True)
