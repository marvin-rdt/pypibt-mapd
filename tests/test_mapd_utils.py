import os

import numpy as np

from pypibt.mapd_utils import (
    Task,
    MAPDInstance,
    parse_mapd_instance,
    get_mapd_instance,
    is_valid_mapd_solution,
    save_mapd_solution,
)
from pypibt.mapf_utils import get_grid


def test_parse_mapd_instance():
    """Test parsing MAPD instance file."""
    config_file = os.path.join(os.path.dirname(__file__), "assets", "test-mapd-config.txt")
    
    map_file, num_agents, task_frequency, task_num, pd_file = parse_mapd_instance(config_file)
    
    # Check parsed values
    assert task_frequency == 1.0
    assert task_num == 5
    assert pd_file is None  # Not specified in test config
    # map_file and num_agents may be None (specified via CLI)


def test_get_mapd_instance():
    """Test getting complete MAPD instance."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    config_file = os.path.join(os.path.dirname(__file__), "assets", "test-mapd-config.txt")
    
    grid, starts, task_frequency, task_num, pickup_locs, delivery_locs, num_agents = get_mapd_instance(
        config_file, 
        map_file=map_name, 
        num_agents=2
    )
    
    # Check grid
    assert grid.shape == (2, 3)
    assert np.array_equal(grid, np.array([[False, True, True], [True, True, True]]))
    
    # Check starts
    assert len(starts) == 2
    assert num_agents == 2
    
    # Check task parameters
    assert task_frequency == 1.0
    assert task_num == 5
    
    # Check pickup/delivery locations (empty when specify_pickup_deliv_locs=0)
    # They will be populated from all free cells when creating MAPDInstance
    assert isinstance(pickup_locs, list)
    assert isinstance(delivery_locs, list)


def test_MAPDInstance():
    """Test MAPD instance class for task generation."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    grid = get_grid(map_name)
    
    pickup_locs = [(0, 1), (0, 2), (1, 0)]
    delivery_locs = [(1, 1), (1, 2)]
    
    instance = MAPDInstance(
        grid=grid,
        task_frequency=1.0,
        task_num=10,
        pickup_locs=pickup_locs,
        delivery_locs=delivery_locs,
        seed=42
    )
    
    # Test task generation using update method
    initial_tasks = len(instance.tasks_unassigned)
    for t in range(20):
        instance.update()
        
        # Check task properties for newly generated tasks
        for task in instance.tasks_unassigned:
            assert task.loc_pickup in pickup_locs
            assert task.loc_delivery in delivery_locs
            assert task.timestep_finished is None
    
    # Should generate task_num tasks total
    total_tasks = len(instance.tasks_unassigned) + len(instance.tasks_assigned) + len(instance.tasks_completed)
    assert total_tasks == 10


def test_Task():
    """Test Task data structure."""
    task = Task(
        id=0,
        loc_pickup=(0, 1),
        loc_delivery=(1, 2),
        loc_current=(0, 1),
        timestep_appear=5,
        timestep_finished=None,
        assigned=False
    )
    
    assert task.id == 0
    assert task.loc_pickup == (0, 1)
    assert task.loc_delivery == (1, 2)
    assert task.loc_current == (0, 1)
    assert task.timestep_appear == 5
    assert task.timestep_finished is None
    assert task.assigned is False


def test_is_valid_mapd_solution():
    """Test MAPD solution validation."""
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    grid = get_grid(map_name)
    starts = [(0, 1), (1, 2)]
    
    # Valid solution with completed task
    solution = [
        [(0, 1), (1, 2)],  # t=0: start
        [(0, 2), (1, 1)],  # t=1: both agents move
        [(1, 2), (1, 0)],  # t=2: continue moving
    ]
    
    # Create a completed task: agent 0 picks up at (0,1) and delivers to (1,2)
    task = Task(
        id=0,
        loc_pickup=(0, 1),
        loc_delivery=(1, 2),
        loc_current=(1, 2),
        timestep_appear=0,
        timestep_finished=2,
        assigned=True
    )
    completed_tasks = [(0, task)]
    
    # Should be valid
    assert is_valid_mapd_solution(grid, starts, solution, completed_tasks)
    
    # Invalid: wrong starts
    assert not is_valid_mapd_solution(grid, [(0, 2), (1, 1)], solution, completed_tasks)
    
    # Invalid: vertex collision
    invalid_solution = [
        [(0, 1), (1, 2)],
        [(0, 2), (0, 2)],  # Both agents at same position
    ]
    assert not is_valid_mapd_solution(grid, starts, invalid_solution, [])
    
    # Invalid: edge collision (swap)
    invalid_solution = [
        [(0, 1), (0, 2)],
        [(0, 2), (0, 1)],  # Agents swap positions
    ]
    assert not is_valid_mapd_solution(grid, starts, invalid_solution, [])


def test_save_mapd_solution():
    """Test saving MAPD solution to file."""
    solution = [
        [(0, 1), (1, 2)],
        [(0, 2), (1, 1)],
        [(1, 2), (1, 0)],
    ]
    
    output_file = os.path.join(os.path.dirname(__file__), "local", "test_save_mapd.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_mapd_solution(solution, output_file)
    
    # Check file exists and has content
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        lines = f.readlines()
        # Should have 3 timesteps with 2 agents each
        assert len(lines) > 0
