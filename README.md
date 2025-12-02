# pypibt-mapd

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](./LICENCE.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of Priority Inheritance with Backtracking (PIBT) for Multi-Agent Path Finding (MAPF) and Multi-Agent Pickup and Delivery (MAPD). This is an **extended version** of [pypibt](https://github.com/Kei18/pypibt).

This repository combines:
- **MAPF implementation**: Unchanged from [pypibt](https://github.com/Kei18/pypibt) by Keisuke Okumura
- **MAPD implementation**: Translated from the C++ version in [pibt2](https://github.com/Kei18/pibt2) by Keisuke Okumura

Both implementations are based on:
- Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2022). Priority inheritance with backtracking for iterative multi-agent path finding. AIJ. [[paper]](https://kei18.github.io/pibt2/)

## Background

PIBT is a powerful algorithm for moving hundreds of agents smoothly in real-time. It was originally developed not just to solve MAPF, but to keep multiple agents running efficiently in dynamic environments.

**MAPF (Multi-Agent Path Finding)**: Find collision-free paths for agents from fixed start positions to fixed goal positions. Use this when you need one-shot path planning.

**MAPD (Multi-Agent Pickup and Delivery)**: Coordinate agents to continuously pick up and deliver items. Tasks appear dynamically, and agents complete them in sequence. Use this for warehouse automation, logistics, and continuous operations.

This Python implementation makes PIBT accessible for:
- Quick prototyping and experimentation
- Integration with Python ML frameworks
- Educational purposes and algorithm understanding
- Extension to new problem domains

## Demo

**MAPF with 100 agents**

<img src="assets/demo_mapf.gif" width="600" alt="MAPF Demo">

**MAPD with 100 agents**

<img src="assets/demo_mapd.gif" width="600" alt="MAPD Demo">

## Setup

This repository is easily setup with [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/marvin-rdt/pypibt-mapd.git
cd pypibt-mapd
uv sync
```

## Usage

### MAPF (Multi-Agent Path Finding)

```sh
# Run MAPF solver
uv run python app.py --mode mapf -m assets/maps/random-32-32-10.map -i assets/mapf/random-32-32-10-random-1.scen -N 100 -o assets/mapf/output_mapf.txt
```

### MAPD (Multi-Agent Pickup and Delivery)

```sh
# Run MAPD solver
uv run python app.py --mode mapd -m assets/maps/random-32-32-10.map -i assets/mapd/task-config.txt -N 100 -o assets/mapd/output_mapd.txt
```

The results will be saved in the output files. The grid maps and scenarios are from [MAPF benchmarks](https://movingai.com/benchmarks/mapf/index.html).

### Visualization

Visualize MAPF and MAPD solutions with the built-in Python visualizer:

```sh
# Visualization of MAPF solution
uv run python scripts/visualize_mapd.py assets/mapf/output_mapf.txt -m assets/maps/random-32-32-10.map --mode mapf

# Visualization of MAPD solution
uv run python scripts/visualize_mapd.py assets/mapd/output_mapd.txt -m assets/maps/random-32-32-10.map --mode mapd

# Save animation as GIF or MP4 with speed control
uv run python scripts/visualize_mapd.py assets/mapd/output_mapd.txt -m assets/maps/random-32-32-10.map --mode mapd --save assets/mapd/output_mapd.gif --speed 5.0

# Fast playback without interpolation
uv run python scripts/visualize_mapd.py assets/mapd/output_mapd.txt -m assets/maps/random-32-32-10.map --mode mapd --no-interpolation
```

**Features:**
- **Agent status**: Agents show loaded (filled) and unloaded (hollow) states
- **Goal lines**: Next goal locations of agents are dynamically visualized
- **Task tracking**: Timestep and completed tasks are tracked (e.g., "Timestep: 50/100 | Tasks: 25/500")
- **Speed control**: `--speed` parameter controls saved animation speed (timesteps/second)
- **Smooth motion**: 5-frame interpolation by default, disable with `--no-interpolation` for faster playback

The MAPF output file is also compatible with the [mapf-visualizer](https://github.com/kei18/mapf-visualizer).

### Jupyter Notebooks

Interactive tutorials with visualization:

```sh
uv run jupyter lab
```

Open these notebooks for step-by-step examples:
- `notebooks/mapf_demo.ipynb` - Two MAPF examples (random and warehouse maps) with animated GIFs
- `notebooks/mapd_demo.ipynb` - Two MAPD examples (random and warehouse maps) with animated GIFs

Both notebooks include complete workflow from problem setup to animated visualization displayed as GIF images (no ffmpeg required).

## License

This software is released under the MIT License, see [LICENSE.txt](LICENCE.txt).

## Citation

If you use this code in academic work, please cite the original paper:

```bibtex
@article{okumura2022priority,
  title={Priority Inheritance with Backtracking for Iterative Multi-agent Path Finding},
  author={Okumura, Keisuke and Machida, Manao and D{\'e}fago, Xavier and Tamura, Yasumasa},
  journal={Artificial Intelligence},
  pages={103752},
  year={2022},
  publisher={Elsevier}
}
```
