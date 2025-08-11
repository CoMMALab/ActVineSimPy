# Contact-Biased Sampling-based Planning as Design for Soft Growing Robots

Yitian Gao<sup>1</sup>\*, Lucas Chen <sup>1</sup>\*, Priyanka Bhovad<sup>2</sup>, Sicheng Wang<sup>2</sup>, Zachary Kingston<sup>1</sup>, and Laura H. Blumenschein<sup>2</sup>

<sup>1</sup> Department of Computer Science, Purdue University

<sup>2</sup> Department of Mechanical Engineering, Purdue University

<bold>*</bold>Equal Contribution

![title](title.gif)

Vine robots are inflatable soft robots that can safely and robustly interact with unstructured and dynamic environments. They can make contact with the environment at no risk (unlike rigid robots), which makes them great for for planning and design optimization tasks.

Previous work has focused on planning under contact for passively deforming vines, but here we investigate *active* steering which is needed for more complex environments. We develop a unified modeling framework that integrates many physical parameters: growth, bending, actuation, and obstacle contact. 

We apply our model in a design optimization task to find designs for vine robots navigating through cluttered environments, identifying designs that minimize the number of required actuators by exploiting environmental contacts. Finally, we fabricate an optimized design and successfully deploy it in an obstacle-rich environment.

# Install

```bash
# If you don't have CUDA capability then remove the jaxlib=*=*cuda*
conda create -n vine jax 'numpy<2' scipy numba matplotlib pandas flax line_profiler "jaxlib=*=*cuda*" jaxopt scikit-learn
conda activate vine

pip install pygame ompl
```

# Run

This repo is divied into multiple parts

- `geometric` The geometric motion planner that finds approximate but dense solutions for the design in order to guide the kinodynamic planner. 
- `kinodynamic` The physically accurate (but slower) planner that uses the vine simulator to explore feasible trajectories.  
- `pbd_vine.py` A JAX vine simulator that finds the forward time evolution of vine states, using Position Based Dynamics to find stable solutions for soft bodies.
- `sPAM` The (inverse) sPAM model, which finds the design parameters to produce a vine that curves at a given angle. Contains a numerical solver and neural surrogate.
- `experiments` Code for experiments; robustness tests, planner comparisons, and plots.  
- `envs` All env files.

For planning

```bash
export PYTHONPATH=$(pwd)
python kinodynamic/sst.py --env envs/divider.txt
```

Runs the geometric planner followed by the kinodynamic planner which jointly find vine designs to solve the give nenv file.  It runs indefinitely in its search for better solutions. Hit ctrl + c to stop it. You can check stats like the minimum solution cost on the top right.

The solutions are saved to `cache/`. View them using

```bash
python kinodynamic/view_solutions.py
```

# Troubleshooting
If you are getting angry yellow warnings about "invalid start state" from rrt, despite the start state being prefectly collision free -- this is not a logic error nor OMPL issue. Instead, go to `_isStateValid` in `kinodynamic/fast_geo/biarc_rrtstar.py`, add some random `print()`, run the code once so Numba recompiles it, then remove the `print()` and it should be good. This is something to do with Numba cache and not a bug that a normal, reasonable person should have to deal with. 

# Common Install Issues
`RuntimeWarning: Your system is avx2 capable but pygame was not built with support for it. The performance of some of your blits could be adversely affected. Consider enabling compile time detection with environment variables like PYGAME_DETECT_AVX2=1 if you are compiling without cross compilation.`
Remove pygame from conda and install via pip instead

