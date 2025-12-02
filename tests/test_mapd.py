import os

import numpy as np

from pypibt import MAPD, is_valid_mapd_solution
from pypibt.mapf_utils import get_grid


def test_MAPD_basic():
    """Test basic MAPD initialization and solving."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    config_file = os.path.join(os.path.dirname(__file__), "assets", "test-mapd-config.txt")
    
    # Initialize MAPD with 2 agents
    mapd = MAPD(config_file, map_file=map_name, num_agents=2, seed=42)
    
    # Check instance properties
    assert mapd.grid.shape == (2, 3)
    assert mapd.N == 2
    assert len(mapd.starts) == 2
    assert mapd.instance.task_num == 5
    
    # Run solver
    solution = mapd.run(max_timestep=100)
    
    # Check solution validity
    assert len(solution) > 0
    assert solution[0] == mapd.starts
    assert is_valid_mapd_solution(mapd.grid, mapd.starts, solution, mapd.completed_tasks)


def test_MAPD_task_completion():
    """Test that MAPD completes tasks successfully."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    config_file = os.path.join(os.path.dirname(__file__), "assets", "test-mapd-config.txt")
    
    mapd = MAPD(config_file, map_file=map_name, num_agents=2, seed=42)
    solution = mapd.run(max_timestep=200)
    
    # Check that some tasks were completed
    assert len(mapd.completed_tasks) > 0
    
    # Check completed task structure
    for agent_id, task in mapd.completed_tasks:
        assert 0 <= agent_id < mapd.N
        assert task.timestep_finished is not None
        # Task can be completed at same timestep if agent starts at pickup location
        assert task.timestep_finished >= task.timestep_appear
        assert task.assigned is True


def test_MAPD_deterministic():
    """Test that MAPD produces deterministic results with same seed."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    config_file = os.path.join(os.path.dirname(__file__), "assets", "test-mapd-config.txt")
    
    # Run with same seed twice
    mapd1 = MAPD(config_file, map_file=map_name, num_agents=2, seed=123)
    solution1 = mapd1.run(max_timestep=100)
    
    mapd2 = MAPD(config_file, map_file=map_name, num_agents=2, seed=123)
    solution2 = mapd2.run(max_timestep=100)
    
    # Solutions should be identical
    assert len(solution1) == len(solution2)
    assert solution1 == solution2
    assert len(mapd1.completed_tasks) == len(mapd2.completed_tasks)


def test_MAPD_max_timestep():
    """Test that MAPD respects max_timestep limit."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    config_file = os.path.join(os.path.dirname(__file__), "assets", "test-mapd-config.txt")
    
    mapd = MAPD(config_file, map_file=map_name, num_agents=2, seed=42)
    max_timestep = 50
    solution = mapd.run(max_timestep=max_timestep)
    
    # Solution should not exceed max_timestep + 1 (includes t=0)
    assert len(solution) <= max_timestep + 1


def test_MAPD_save_solution():
    """Test saving MAPD solution to file."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    config_file = os.path.join(os.path.dirname(__file__), "assets", "test-mapd-config.txt")
    
    mapd = MAPD(config_file, map_file=map_name, num_agents=2, seed=42)
    solution = mapd.run(max_timestep=100)
    
    # Save solution
    output_file = os.path.join(os.path.dirname(__file__), "local", "test_mapd_output.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mapd.save_solution(output_file)
    
    # Check file was created
    assert os.path.exists(output_file)
    
    # Check file has content
    with open(output_file, 'r') as f:
        content = f.read()
        assert len(content) > 0
