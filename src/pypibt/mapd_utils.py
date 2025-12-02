"""Utilities and data structures for Multi-Agent Pickup and Delivery (MAPD).

This module provides the core data structures and utilities for MAPD problems,
including task representation, dynamic task generation, instance parsing, solution
validation, and output formatting.

References:
    Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2022).
    Priority inheritance with backtracking for iterative multi-agent
    path finding. Artificial Intelligence Journal.
    https://kei18.github.io/pibt2/
"""

from dataclasses import dataclass
import random

from .mapf_utils import Coord, Grid, Config, Configs, get_neighbors


@dataclass
class Task:
    """Pickup and delivery task for MAPD.
    
    Represents a single task requiring an agent to pick up an item from one
    location and deliver it to another. Tasks appear dynamically during
    execution and track their lifecycle from appearance to completion.
    
    Attributes:
        id: Unique task identifier (0, 1, 2, ...).
        loc_pickup: Pickup location coordinates (y, x).
        loc_delivery: Delivery location coordinates (y, x).
        loc_current: Current item location (pickup initially, then with agent).
        timestep_appear: Timestep when task became available.
        timestep_finished: Timestep when task was completed (None if ongoing).
        assigned: True if task has been assigned to an agent.
    
    Lifecycle:
        1. Task appears (assigned=False, loc_current=loc_pickup)
        2. Agent targets and reaches pickup (assigned=True)
        3. Agent moves with item (loc_current tracks agent position)
        4. Agent reaches delivery (timestep_finished set, task completed)
    """
    id: int
    loc_pickup: Coord
    loc_delivery: Coord
    loc_current: Coord
    timestep_appear: int
    timestep_finished: int | None
    assigned: bool
    
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.timestep_finished is not None


class MAPDInstance:
    """MAPD problem instance with dynamic task generation.
    
    Manages the lifecycle of pickup and delivery tasks in a MAPD problem.
    Tasks are generated dynamically based on frequency parameters, appear at
    specified timesteps, and are tracked through completion.
    
    The instance maintains three task lists:
    - tasks_unassigned: Tasks waiting to be picked up (not yet assigned)
    - tasks_assigned: Tasks currently being carried by agents
    - tasks_completed: Tasks that have been delivered
    
    Task Generation:
        - task_frequency < 1: Probabilistic (e.g., 0.5 = 50% chance per timestep)
        - task_frequency >= 1: Deterministic (e.g., 2.0 = every 2 timesteps)
    
    Attributes:
        grid: 2D boolean array representing the map.
        task_frequency: Rate of task generation (see above).
        task_num: Total number of tasks to generate across entire run.
        seed: Random seed for reproducible task generation.
        current_timestep: Current simulation timestep.
        tasks_unassigned: List of tasks available for pickup.
        tasks_assigned: List of tasks being carried by agents.
        tasks_completed: List of delivered tasks.
        locs_pickup: Valid pickup locations on the map.
        locs_delivery: Valid delivery locations on the map.
    """
    
    def __init__(self, grid: Grid, task_frequency: float, task_num: int,
                 pickup_locs: list[Coord], delivery_locs: list[Coord], seed: int = 0):
        """Initialize MAPD instance with task parameters.
        
        Args:
            grid: 2D boolean array where True indicates free space.
            task_frequency: Task generation rate. If < 1, probabilistic chance
                          per timestep. If >= 1, deterministic interval.
            task_num: Total number of tasks to generate over entire run.
            pickup_locs: List of valid pickup coordinates. If empty, uses all free cells.
            delivery_locs: List of valid delivery coordinates. If empty, uses all free cells.
            seed: Random seed for deterministic task generation (default: 0).
        """
        self.grid = grid
        self.task_frequency = task_frequency
        self.task_num = task_num
        self.seed = seed
        
        self.current_timestep = 0
        self.tasks_unassigned: list[Task] = []  # Tasks available for pickup
        self.tasks_assigned: list[Task] = []    # Tasks being carried
        self.tasks_completed: list[Task] = []   # Delivered tasks
        
        # Endpoint locations (filter out dead ends if not provided)
        if pickup_locs:
            self.locs_pickup = self._filter_dead_ends(pickup_locs)
        else:
            self.locs_pickup = self._get_valid_free_cells()
            
        if delivery_locs:
            self.locs_delivery = self._filter_dead_ends(delivery_locs)
        else:
            self.locs_delivery = self._get_valid_free_cells()
        
        # Validate we have sufficient locations
        if len(self.locs_pickup) == 0:
            raise ValueError("No valid pickup locations (all are dead ends)")
        if len(self.locs_delivery) == 0:
            raise ValueError("No valid delivery locations (all are dead ends)")
        
        # Random number generator
        self.rng = random.Random(seed)
        
        # Task generation state
        self._next_task_id = 0
        self._tasks_generated = 0
    
    def _filter_dead_ends(self, locations: list[Coord]) -> list[Coord]:
        """Filter out dead end positions from location list.
        
        Args:
            locations: List of coordinates to filter.
        
        Returns:
            List of coordinates that are not dead ends (have >= 2 neighbors).
        """
        from .mapf_utils import is_dead_end
        filtered = [loc for loc in locations if not is_dead_end(self.grid, loc)]
        
        if len(filtered) < len(locations):
            print(f"Info: Filtered {len(locations) - len(filtered)} dead end locations "
                  f"from task endpoints")
        
        return filtered
    
    def _get_valid_free_cells(self) -> list[Coord]:
        """Get all valid free cells (not dead ends) from grid.
        
        Returns:
            List of coordinates with at least 2 free neighbors.
        """
        from .mapf_utils import get_valid_positions
        return get_valid_positions(self.grid, min_neighbors=2)
    
    def _get_all_free_cells(self) -> list[Coord]:
        """Get all free cells from grid."""
        free_cells = []
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x]:
                    free_cells.append((y, x))
        return free_cells
    
    def update(self) -> None:
        """Update instance state (called each timestep).
        
        Increments timestep and generates new tasks according to task_frequency.
        """
        self.current_timestep += 1
        self._generate_tasks_for_timestep()
    
    def _generate_tasks_for_timestep(self) -> None:
        """Generate new tasks for current timestep.
        
        Generation strategy:
        - If task_frequency >= 1: Generate that many tasks per timestep
        - If task_frequency < 1: Probabilistic generation
        """
        if self._tasks_generated >= self.task_num:
            return  # Already generated all tasks
        
        # Determine how many tasks to generate
        if self.task_frequency >= 1.0:
            # Deterministic: Generate fixed number per timestep
            num_to_gen = int(self.task_frequency)
        else:
            # Probabilistic: Generate with probability
            num_to_gen = 1 if self.rng.random() < self.task_frequency else 0
        
        # Generate tasks
        for _ in range(num_to_gen):
            if self._tasks_generated >= self.task_num:
                break
            
            # Random pickup and delivery
            pickup = self.rng.choice(self.locs_pickup)
            delivery = self.rng.choice(self.locs_delivery)
            
            # Ensure pickup != delivery
            while pickup == delivery and len(self.locs_delivery) > 1:
                delivery = self.rng.choice(self.locs_delivery)
            
            task = Task(
                id=self._next_task_id,
                loc_pickup=pickup,
                loc_delivery=delivery,
                loc_current=pickup,
                timestep_appear=self.current_timestep,
                timestep_finished=None,
                assigned=False
            )
            
            # print(f"[t={self.current_timestep}] Task {task.id} generated: pickup=({pickup[1]},{pickup[0]}) -> delivery=({delivery[1]},{delivery[0]})")
            
            self.tasks_unassigned.append(task)
            self._next_task_id += 1
            self._tasks_generated += 1


def parse_mapd_instance(instance_file: str) -> tuple[str | None, int, float, int, 
                                                      str | None]:
    """Parse MAPD instance file.
    
    Supports two formats:
    1. Full format (legacy): includes map_file=...
    2. Simplified format: omits map_file (specified via command line)
    
    Format:
        map_file=path/to/map.map (optional, for legacy compatibility)
        agents=N (optional if map not included)
        task_frequency=<float>
        task_num=<int>
        pd_file=path/to/pickups_deliveries.pd (optional)
    
    Note: seed and max_timestep are NOT in config file - passed via command line like MAPF
    Note: All parameters have defaults defined in task-config.txt
    
    Args:
        instance_file: Path to instance file.
    
    Returns:
        Tuple of (map_file, num_agents, task_frequency, task_num, pd_file).
        Note: map_file and num_agents may be None in simplified format.
    """
    import os
    
    # Values that can come from config file
    map_file = None
    num_agents = 0
    task_frequency = None
    task_num = None
    pd_file = None
    
    instance_dir = os.path.dirname(os.path.abspath(instance_file))
    
    with open(instance_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'map_file':
                    map_file = os.path.join(instance_dir, value) if not os.path.isabs(value) else value
                elif key == 'agents':
                    # Parse number of agents
                    num_agents = int(value)
                elif key == 'task_frequency':
                    task_frequency = float(value)
                elif key == 'task_num':
                    task_num = int(value)
                elif key == 'pd_file':
                    pd_file = os.path.join(instance_dir, value) if not os.path.isabs(value) else value
    
    # Validate required parameters from config file
    if task_frequency is None:
        raise ValueError("task_frequency must be specified in config file")
    if task_num is None:
        raise ValueError("task_num must be specified in config file")
    
    # Note: map_file and num_agents may be None (supplied via command line)
    # Note: seed and max_timestep are NOT parsed from file - passed via command line argument
    return map_file, num_agents, task_frequency, task_num, pd_file


def parse_pd_file(grid: Grid, pd_file: str) -> tuple[list[Coord], list[Coord]]:
    """Parse pickup/delivery location file.
    
    Format: Grid with special characters:
        'p' = pickup only
        'd' = delivery only
        's' = both pickup and delivery (service point)
        'e' = endpoint (neither pickup nor delivery)
        'a' = all (pickup, delivery, and endpoint)
    
    Args:
        grid: Map grid.
        pd_file: Path to pd file.
    
    Returns:
        Tuple of (pickup_locs, delivery_locs).
    """
    import os
    
    if not os.path.exists(pd_file):
        return [], []
    
    pickup_locs = []
    delivery_locs = []
    
    with open(pd_file, 'r') as f:
        for y, line in enumerate(f):
            line = line.rstrip('\r\n')
            for x, char in enumerate(line):
                # Check if location is valid
                if y >= grid.shape[0] or x >= grid.shape[1]:
                    continue
                if not grid[y, x]:
                    continue
                
                # Parse special characters
                if char in ['p', 's', 'a']:
                    pickup_locs.append((y, x))
                if char in ['d', 's', 'a']:
                    delivery_locs.append((y, x))
    
    return pickup_locs, delivery_locs


def get_mapd_instance(instance_file: str, map_file: str | None = None, 
                      num_agents: int | None = None, seed: int = 0) -> \
        tuple[Grid, list[Coord], float, int, list[Coord], list[Coord], int]:
    """Load complete MAPD instance from file.
    
    Validates that agent starts, pickup locations, and delivery locations are not
    in dead ends (have at least 2 free neighbors) to prevent PIBT deadlocks.
    
    Args:
        instance_file: Path to MAPD instance file.
        map_file: Path to map file (overrides value in instance_file if provided).
        num_agents: Number of agents (overrides value in instance_file if provided).
        seed: Random seed for agent start positions and task generation (from command line).
    
    Returns:
        Tuple of (grid, starts, task_frequency, task_num, pickup_locs, delivery_locs,
                  num_agents).
    
    Raises:
        ValueError: If insufficient valid (non-dead-end) positions for agents/tasks.
    """
    from .mapf_utils import get_grid, get_valid_positions, is_dead_end
    import random
    
    # Parse instance file
    map_file_from_file, num_agents_from_file, task_frequency, task_num, pd_file = \
        parse_mapd_instance(instance_file)
    
    # Use provided values or fall back to file values
    final_map_file = map_file if map_file is not None else map_file_from_file
    final_num_agents = num_agents if num_agents is not None else num_agents_from_file
    
    # Note: seed comes from command line argument, not config file
    
    if final_map_file is None:
        raise ValueError("map_file must be specified either in instance file or as argument")
    if final_num_agents is None or final_num_agents == 0:
        raise ValueError("num_agents must be specified either in instance file or as argument")
    
    # Load grid
    grid = get_grid(final_map_file)
    
    # Get all valid positions (not dead ends - have at least 2 neighbors)
    valid_positions = get_valid_positions(grid, min_neighbors=2)
    
    if len(valid_positions) < final_num_agents:
        raise ValueError(
            f"Insufficient valid positions (not dead ends): {len(valid_positions)} available, "
            f"{final_num_agents} agents required. Dead ends have <2 neighbors and cause PIBT deadlocks."
        )
    
    # Generate random start positions from valid positions only
    rng = random.Random(seed)
    starts = rng.sample(valid_positions, final_num_agents)
    
    # Validate starts
    for i, start in enumerate(starts):
        if is_dead_end(grid, start):
            print(f"Warning: Agent {i} start position {start} is in a dead end!")
    
    # Load pickup/delivery locations
    pickup_locs = []
    delivery_locs = []
    if pd_file:
        pickup_locs, delivery_locs = parse_pd_file(grid, pd_file)
        
        # Filter out dead ends from pickup/delivery locations
        from .mapf_utils import is_dead_end
        pickup_locs_filtered = [loc for loc in pickup_locs if not is_dead_end(grid, loc)]
        delivery_locs_filtered = [loc for loc in delivery_locs if not is_dead_end(grid, loc)]
        
        # Warn if any were filtered
        if len(pickup_locs_filtered) < len(pickup_locs):
            print(f"Warning: Filtered {len(pickup_locs) - len(pickup_locs_filtered)} "
                  f"pickup locations in dead ends")
        if len(delivery_locs_filtered) < len(delivery_locs):
            print(f"Warning: Filtered {len(delivery_locs) - len(delivery_locs_filtered)} "
                  f"delivery locations in dead ends")
        
        pickup_locs = pickup_locs_filtered
        delivery_locs = delivery_locs_filtered
        
        # Validate sufficient locations
        if len(pickup_locs) == 0:
            raise ValueError("No valid pickup locations after filtering dead ends")
        if len(delivery_locs) == 0:
            raise ValueError("No valid delivery locations after filtering dead ends")
    
    return grid, starts, task_frequency, task_num, pickup_locs, delivery_locs, \
           final_num_agents


def is_valid_mapd_solution(grid: Grid, starts: list[Coord], solution: Configs,
                            completed_tasks: list[tuple[int, Task]]) -> bool:
    """Validate MAPD solution.
    
    Checks:
        1. No vertex collisions (two agents at same position)
        2. No edge collisions (two agents swapping positions)
        3. All moves are valid (agent moves to neighbor or stays)
        4. For each completed task: agent visited pickup before delivery
    
    Args:
        grid: Map grid.
        starts: Initial agent positions.
        solution: Sequence of configurations.
        completed_tasks: List of (agent_id, task) pairs.
    
    Returns:
        True if solution is valid, False otherwise.
    """
    if not solution:
        return False
    
    # Check initial configuration
    if solution[0] != starts:
        return False
    
    # Check all timesteps for collisions and valid moves
    T = len(solution)
    N = len(starts)
    
    for t in range(T):
        config = solution[t]
        
        for i in range(N):
            v_i_now = config[i]
            
            if t > 0:
                v_i_pre = solution[t - 1][i]
                
                # Check continuity (agent can only move to neighbors or stay)
                valid_moves = [v_i_pre] + get_neighbors(grid, v_i_pre)
                if v_i_now not in valid_moves:
                    return False
            
            # Check vertex collisions with other agents
            for j in range(i + 1, N):
                v_j_now = config[j]
                
                # Vertex collision: two agents at same position
                if v_i_now == v_j_now:
                    return False
                
                # Edge collision (swap): check if agents swapped positions
                if t > 0:
                    v_j_pre = solution[t - 1][j]
                    # Agent i moved from v_i_pre to v_i_now
                    # Agent j moved from v_j_pre to v_j_now  
                    # Collision if i moved to j's old position AND j moved to i's old position
                    if v_i_now == v_j_pre and v_j_now == v_i_pre:
                        return False
    
    # Check each completed task: agent must visit pickup before delivery
    for agent_id, task in completed_tasks:
        # Find first timestep when agent was at pickup
        t_pickup = None
        # Find last timestep when agent was at delivery  
        t_delivery = None
        
        for t, config in enumerate(solution):
            if config[agent_id] == task.loc_pickup:
                if t_pickup is None:
                    t_pickup = t
            if config[agent_id] == task.loc_delivery:
                t_delivery = t
        
        # Verify pickup happened before delivery
        if t_pickup is None or t_delivery is None or t_pickup >= t_delivery:
            return False
    
    return True


def save_mapd_solution(solution: Configs, output_file: str, 
                        agent_goals: list[list[tuple[Coord, str]]] = None) -> None:
    """Save MAPD solution with agent goals and status at each timestep.
    
    Format: 
        timestep:(x,y)[gx,gy,status],(x,y)[gx,gy,status],...
        where status is: 'A' (assigned/carrying), 'T' (targeting pickup), 'F' (free)
    
    Followed by agent state documentation:
        # Agent States per Timestep
        t,agent_id,pos_x,pos_y,goal_x,goal_y,status
    
    Args:
        solution: Sequence of configurations (agent positions).
        output_file: Output file path.
        agent_goals: Optional list of per-timestep goals and status: [[(goal, status), ...], ...]
    """
    with open(output_file, 'w') as f:
        # Write agent trajectories with goals and status
        for t, config in enumerate(solution):
            f.write(f"{t}:")
            parts = []
            for i, v in enumerate(config):
                pos_str = f"({v[1]},{v[0]})"
                if agent_goals and t < len(agent_goals) and i < len(agent_goals[t]):
                    goal, status = agent_goals[t][i]
                    goal_str = f"[{goal[1]},{goal[0]},{status}]"
                    parts.append(pos_str + goal_str)
                else:
                    parts.append(pos_str)
            f.write(','.join(parts))
            f.write(',\n')
        
        # Write detailed agent state documentation
        if agent_goals:
            f.write('\n# Agent States per Timestep\n')
            f.write('# Format: timestep,agent_id,pos_x,pos_y,goal_x,goal_y,status\n')
            for t, config in enumerate(solution):
                if t < len(agent_goals):
                    for i, v in enumerate(config):
                        if i < len(agent_goals[t]):
                            goal, status = agent_goals[t][i]
                            f.write(f"{t},{i},{v[1]},{v[0]},{goal[1]},{goal[0]},{status}\n")
