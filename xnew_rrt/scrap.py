import os


PATH = "xnew_rrt/sim_animation"
for root, dirs, files in os.walk(PATH, topdown=False):
    for name in files:
        if name == "final_gif.gif":
            os.remove(os.path.join(root, name))
    for name in dirs:
        full = os.path.join(root, name)
        if os.path.islink(full):
            os.remove(full)   # symlink to a dir: remove the link itself
        else:
            os.rmdir(full)