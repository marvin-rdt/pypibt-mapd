"""Priority Inheritance with Backtracking (PIBT) algorithm for MAPD.

This module implements Multi-Agent Pickup and Delivery (MAPD) using the PIBT
algorithm. PIBT provides fast, collision-free coordination for agents that
must dynamically pick up and deliver items.

Key Differences from MAPF:
    - Dynamic goal assignment: agents switch between pickup and delivery goals
    - Task-based priorities: agents carrying items have higher priority
    - Continuous operation: agents complete tasks and receive new ones
    - Termination: based on task completion rather than fixed goals

References:
    Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2022).
    Priority inheritance with backtracking for iterative multi-agent
    path finding. Artificial Intelligence Journal.
    https://kei18.github.io/pibt2/
"""

from dataclasses import dataclass

import numpy as np

from .dist_table import DistTable
from .mapd_utils import Task, MAPDInstance, get_mapd_instance, save_mapd_solution
from .mapf_utils import Config, Configs, Coord, Grid, get_neighbors


@dataclass
class Agent:
    """Agent representation for MAPD with dynamic goal assignment.
    
    Unlike MAPF agents with fixed goals, MAPD agents dynamically switch between
    targeting pickup locations, carrying items to delivery locations, and waiting
    for new tasks. Priority is determined by task status, elapsed time, and a
    random tie-breaker.
    
    Attributes:
        id: Unique agent identifier (0 to N-1).
        v_now: Current position (y, x) coordinates.
        v_next: Planned next position, None if planning not yet completed.
        g: Current goal position (changes dynamically with tasks).
        elapsed: Time steps since last goal reached (used for priority).
        tie_breaker: Random value in [0,1) for deterministic tie-breaking.
        task: Currently assigned task (agent is carrying item to delivery).
        target_task: Task being targeted (agent is moving to pickup location).
    
    Task States:
        - Assigned (task is not None): Agent has picked up item, moving to delivery
        - Targeting (target_task is not None): Agent is moving to pickup location
        - Free (both None): Agent is idle, waiting for task assignment
    """
    id: int
    v_now: Coord
    v_next: Coord | None
    g: Coord
    elapsed: int
    tie_breaker: float
    task: Task | None
    target_task: Task | None
    
    def is_assigned(self) -> bool:
        """Check if agent has an assigned task (carrying item)."""
        return self.task is not None
    
    def is_targeting(self) -> bool:
        """Check if agent is targeting a pickup (moving to pickup)."""
        return self.target_task is not None
    
    def is_free(self) -> bool:
        """Check if agent is free (no task or target)."""
        return self.task is None and self.target_task is None


class MAPD:
    """Priority Inheritance with Backtracking for Multi-Agent Pickup and Delivery.
    
    MAPD extends PIBT to handle dynamic pickup and delivery tasks. Agents continuously
    receive tasks, navigate to pickup locations, pick up items, deliver them, and
    receive new tasks. The algorithm ensures collision-free paths while prioritizing
    agents carrying items over idle agents.
    
    Task Lifecycle:
        1. Task appears in environment (pickup and delivery locations)
        2. Free agent targets nearest unassigned pickup location
        3. Upon reaching pickup, agent receives task assignment
        4. Agent navigates to delivery location with item
        5. Upon reaching delivery, task completes and agent becomes free
        6. Cycle repeats with new task assignment
    
    Priority System:
        Agents are prioritized as: assigned (carrying item) > elapsed time > tie-breaker
        This ensures agents with items complete deliveries before idle agents claim new tasks.
    
    Attributes:
        grid: 2D boolean array where True indicates free space.
        starts: Initial positions of all agents.
        N: Number of agents in the system.
        seed: Random seed for reproducible tie-breaking.
        rng: Random number generator for tie-breaking.
        instance: MAPD instance manager (generates tasks dynamically).
        dist_tables_cache: Cached distance tables for each unique goal.
        NIL: Sentinel value representing unassigned agent.
        occupied_now: Current occupation status of each grid cell.
        occupied_next: Next timestep occupation status (for planning).
        solution: Complete trajectory of all agents over time.
        completed_tasks: List of (agent_id, task) pairs for finished tasks.
    
    Example:
        >>> mapd = MAPD("instance.txt", seed=42)
        >>> solution = mapd.run(max_timestep=1000)
        >>> print(f"Completed {len(mapd.completed_tasks)} tasks in {len(solution)} steps")
    
    References:
        Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2022).
        Priority inheritance with backtracking for iterative multi-agent
        path finding. Artificial Intelligence Journal.
        https://kei18.github.io/pibt2/
    """
    
    def __init__(self, instance_file: str, map_file: str | None = None, 
                 num_agents: int | None = None, seed: int = 0):
        """Initialize MAPD solver with problem instance.
        
        Args:
            instance_file: Path to MAPD instance file containing task configuration.
            map_file: Path to map file (optional, can be in instance_file for legacy compatibility).
            num_agents: Number of agents (optional, can be in instance_file for legacy compatibility).
            seed: Random seed for deterministic tie-breaking and task generation (default: 0).
        """
        # Load instance (seed passed to get_mapd_instance for start position generation)
        (grid, starts, task_frequency, task_num, pickup_locs, 
         delivery_locs, num_agents) = get_mapd_instance(
            instance_file, map_file, num_agents, seed
        )
        
        self.grid = grid
        self.starts = starts
        self.N = num_agents
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # MAPD instance manager (handles task generation)
        # Uses same seed for task generation to ensure reproducibility
        self.instance = MAPDInstance(
            grid, task_frequency, task_num,
            pickup_locs, delivery_locs, seed
        )
        
        # Distance tables cache (goal -> DistTable)
        # Each unique goal gets its own distance table
        # Use 2D numpy array for faster indexing (vs dict with tuple keys)
        self.dist_tables_cache: np.ndarray = np.empty(self.grid.shape, dtype=object)
        
        # Pre-compute neighbor lists for all positions (avoid repeated computation)
        self.neighbors_cache: dict[Coord, list[Coord]] = {}
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x]:  # Only cache for valid positions
                    coord = (y, x)
                    self.neighbors_cache[coord] = get_neighbors(self.grid, coord)
        
        # Reservation tables for PIBT
        self.NIL = self.N  # Sentinel value for "empty"
        self.occupied_now = np.full(self.grid.shape, self.NIL, dtype=int)
        self.occupied_next = np.full(self.grid.shape, self.NIL, dtype=int)
        
        # Solution tracking
        self.solution: Configs = []
        self.completed_tasks: list[tuple[int, Task]] = []  # (agent_id, task)
        self.agent_goals_history: list[list[tuple[Coord, str]]] = []  # Per-timestep [(goal, status), ...]
    
    def run(self, max_timestep: int = 1000, max_comp_time: int = 60000) -> Configs:
        """Execute MAPD simulation until all tasks complete or timeout.
        
        Runs the main MAPD loop coordinating task assignment, collision-free
        planning via PIBT, execution, and task lifecycle management. Continues
        until all tasks are completed, maximum timesteps reached, or computation
        time limit exceeded.
        
        Main Loop Phases:
            1. Task Assignment: Assign nearest unassigned pickups to free agents
            2. Planning: Apply PIBT algorithm with priority-based ordering
            3. Execution: Move agents to planned positions
            4. Task Updates: Handle pickup/delivery events
            5. Environment: Generate new tasks, increment timestep
        
        Args:
            max_timestep: Maximum simulation timesteps (default: 1000).
            max_comp_time: Maximum computation time in milliseconds (default: 60000).
        
        Returns:
            Complete solution as sequence of configurations. Each configuration
            is a list of (y,x) positions for all agents at that timestep.
            
        Example:
            >>> mapd = MAPD("warehouse.txt")
            >>> solution = mapd.run(max_timestep=500, max_comp_time=30000)
            >>> save_mapd_solution("output.txt", solution, mapd.completed_tasks)
        """
        import time
        start_time = time.perf_counter()
        max_comp_time_sec = max_comp_time / 1000.0
        
        # Initialize agents
        agents = self._initialize_agents()
        
        # Add initial configuration and agent states
        self.solution.append(self.starts)
        self._record_agent_goals(agents)
        
        # Initialize environment (timestep 0, generate initial tasks)
        self.instance.update()
        
        # Main simulation loop
        while self.instance.current_timestep < max_timestep:
            # Check computation time limit
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time >= max_comp_time_sec:
                print(f"Computation time limit reached: {elapsed_time:.2f}s")
                break
            
            # === PHASE 1: TASK ASSIGNMENT ===
            self._assign_tasks(agents)
            
            # === PHASE 2: PLANNING (PIBT) ===
            self._plan_agents(agents)
            
            # === PHASE 3: ACTING ===
            config = self._execute_actions(agents)
            self.solution.append(config)
            
            # === PHASE 4: TASK STATE UPDATE ===
            # Process task completions/pickups at NEW positions
            self._update_task_states(agents)
            
            # Record agent goals AFTER task updates (final state for this timestep)
            self._record_agent_goals(agents)
            
            # === PHASE 5: ENVIRONMENT UPDATE ===
            self.instance.update()
            
            # Check termination
            if len(self.instance.tasks_completed) >= self.instance.task_num:
                break  # Success: all tasks completed
        
        # Check if max timesteps reached
        if self.instance.current_timestep >= max_timestep:
            print(f"Maximum timestep limit reached: {max_timestep}")
        
        return self.solution
    
    def _record_agent_goals(self, agents: list[Agent]) -> None:
        """Record current goals and status for all agents.
        
        Agents must be recorded in ID order to match the solution trajectory order.
        """
        # Sort by agent ID to ensure consistent ordering
        sorted_agents = sorted(agents, key=lambda a: a.id)
        
        goals_and_status = []
        for agent in sorted_agents:
            if agent.is_free():
                status = 'F'  # Free
            elif agent.is_assigned():
                status = 'A'  # Assigned/carrying
            elif agent.is_targeting():
                status = 'T'  # Targeting pickup
            else:
                status = 'F'  # Free (fallback)
            goals_and_status.append((agent.g, status))
        self.agent_goals_history.append(goals_and_status)
    
    def _initialize_agents(self) -> list[Agent]:
        """Initialize all agents at their starting positions.
        
        Creates Agent objects for each starting position with initial state:
        - Goal set to current position (free agent)
        - No assigned or target tasks
        - Random tie-breaker for priority resolution
        - Marks starting positions as occupied
        
        Returns:
            List of initialized Agent objects.
        """
        agents = []
        for i, start in enumerate(self.starts):
            agent = Agent(
                id=i,
                v_now=start,
                v_next=None,
                g=start,  # Goal starts at current position (free agent)
                elapsed=0,
                tie_breaker=self.rng.random(),
                task=None,
                target_task=None
            )
            agents.append(agent)
            self.occupied_now[start] = i
        return agents
    
    def _assign_tasks(self, agents: list[Agent]) -> None:
        """Phase 1: Assign target tasks to free agents.
        
        Assigns each free agent to target the nearest unassigned and untargeted
        pickup location. Once an agent starts targeting a task, it sticks with
        that task until either:
        1. It successfully picks up the task (reaches pickup location)
        2. Another agent picks up the task first (task becomes assigned)
        
        This prevents task switching and ensures stable agent behavior.
        
        Args:
            agents: List of all agents to process.
        """
        # Get tasks that are not assigned and not already being targeted
        # Use set of task IDs for O(1) membership testing
        targeted_task_ids = {agent.target_task.id for agent in agents 
                            if agent.target_task is not None}
        unassigned_task_ids = {t.id for t in self.instance.tasks_unassigned}
        available_tasks = [t for t in self.instance.tasks_unassigned 
                          if t.id not in targeted_task_ids]
        
        # Shuffle for randomness in tie-breaking
        self.rng.shuffle(available_tasks)
        
        for agent in agents:
            # Skip agents that already have assigned tasks (carrying items)
            if agent.is_assigned():
                # Goal must remain locked to delivery location - never change it
                assert agent.g == agent.task.loc_delivery, \
                    f"Agent {agent.id} has assigned task but goal is not delivery location!"
                continue
            
            # If agent is already targeting a task, keep it unless task is no longer valid
            if agent.is_targeting():
                # Check if target is still valid (not assigned by someone else)
                # Use set for O(1) membership check
                if agent.target_task.id not in unassigned_task_ids:
                    # Task was taken by another agent - need new target
                    agent.target_task = None
                    agent.g = agent.v_now
                else:
                    # Keep current target, goal stays the same
                    continue
            
            # Agent is free or lost its target - find nearest available pickup
            if not agent.is_targeting():
                # Default: stay at current location (free agents should not wander)
                agent.g = agent.v_now
                
                # Only search for tasks if available
                if not available_tasks:
                    # No tasks available - stay put with lowest priority
                    agent.elapsed = 0  # Reset to maintain lowest priority among free agents
                    continue
                
                min_dist = self.grid.size
                best_task = None
                
                for task in available_tasks:
                    dist = self._get_path_dist(agent.v_now, task.loc_pickup)
                    
                    # Special case: agent is already at pickup location
                    if dist == 0:
                        assert agent.v_now == task.loc_pickup, \
                            f"Agent {agent.id} at {agent.v_now} but pickup is {task.loc_pickup}, dist={dist}!"
                        # Move task from unassigned to assigned
                        self.instance.tasks_unassigned.remove(task)
                        self.instance.tasks_assigned.append(task)
                        self._assign_task(agent, task)
                        available_tasks.remove(task)
                        best_task = None  # Already assigned, don't set as target
                        break  # Task assigned, stop searching
                    
                    # Track closest available task
                    if dist < min_dist:
                        min_dist = dist
                        best_task = task
                
                # Set target and goal for best task found
                if best_task is not None:
                    agent.target_task = best_task
                    agent.g = best_task.loc_pickup
                    available_tasks.remove(best_task)  # Mark as targeted
    
    def _assign_task(self, agent: Agent, task: Task) -> None:
        """Officially assign task to agent upon reaching pickup location.
        
        Transitions agent from targeting state to assigned state. Updates
        agent's goal from pickup to delivery location and marks task as assigned
        to prevent other agents from claiming it.
        
        Args:
            agent: Agent receiving the task assignment.
            task: Task being assigned (must not already be assigned).
        """
        agent.task = task
        agent.target_task = None
        task.assigned = True
        agent.g = task.loc_delivery  # Update goal to delivery location
    
    def _plan_agents(self, agents: list[Agent]) -> None:
        """Phase 2: Compute collision-free next positions using PIBT.
        
        Applies the Priority Inheritance with Backtracking algorithm to plan
        the next position for each agent. Agents are processed in priority order
        (assigned > elapsed > tie-breaker) to ensure agents carrying items have
        precedence.
        
        The PIBT algorithm ensures no vertex or edge collisions through recursive
        priority inheritance and backtracking when conflicts arise.
        
        Args:
            agents: List of all agents (modified in-place with v_next positions).
        """
        # Create agent lookup using list for O(1) access by ID
        # Assumes agent IDs are sequential starting from 0
        agent_list = [None] * len(agents)
        for agent in agents:
            agent_list[agent.id] = agent
        
        # Sort by priority (highest first)
        agents.sort(key=self._agent_priority_key, reverse=True)
        
        # Plan each agent that hasn't been planned yet
        for agent in agents:
            if agent.v_next is None:
                self._func_pibt(agent, agent_list)
    
    def _execute_actions(self, agents: list[Agent]) -> Config:
        """Phase 3: Execute planned moves and update agent states.
        
        Moves all agents to their planned next positions (v_next -> v_now),
        updates reservation tables, and recalculates priority counters.
        Agents at their goals have elapsed time reset to 0, others increment.
        
        Args:
            agents: List of all agents with planned v_next positions.
        
        Returns:
            Configuration (list of positions) sorted by agent ID for output.
        """
        for agent in agents:
            # Clear reservation tables (no conditional needed - we know agent owns these cells)
            self.occupied_now[agent.v_now] = self.NIL
            if agent.v_next is not None:
                self.occupied_next[agent.v_next] = self.NIL
            
            # Execute move
            agent.v_now = agent.v_next
            self.occupied_now[agent.v_now] = agent.id
            
            # Update priority counter
            # Reset to 0 if at goal, otherwise increment
            agent.elapsed = 0 if agent.v_now == agent.g else agent.elapsed + 1
            
            # Reset planning
            agent.v_next = None
        
        # Return configuration sorted by agent ID
        agents_sorted = sorted(agents, key=lambda a: a.id)
        config = [agent.v_now for agent in agents_sorted]
        return config
    
    def _update_task_states(self, agents: list[Agent]) -> None:
        """Phase 4: Update task lifecycle based on agent positions.
        
        Processes task state transitions for all agents:
        - Assigned agents: Check if delivery location reached (task completion)
        - Targeting agents: Check if pickup location reached (task assignment)
        
        Only the first agent to reach a pickup location will be assigned the task.
        Other agents targeting the same location will have their target cleared
        in the next assignment phase.
        
        Args:
            agents: List of all agents to process.
        """
        for agent in agents:
            if agent.is_assigned():
                # Agent has assigned task - goal must be delivery location
                assert agent.g == agent.task.loc_delivery, \
                    f"Agent {agent.id} carrying task but goal changed from delivery!"
                
                # Update task's current location to agent position
                agent.task.loc_current = agent.v_now
                
                if agent.v_now == agent.task.loc_delivery:
                    # Task completed!
                    agent.task.timestep_finished = self.instance.current_timestep
                    
                    # Move task from assigned to completed
                    self.instance.tasks_assigned.remove(agent.task)
                    self.instance.tasks_completed.append(agent.task)
                    self.completed_tasks.append((agent.id, agent.task))
                    
                    # Agent becomes free immediately with lowest priority
                    # Goal set to current position so agent stays put and doesn't block others
                    agent.task = None
                    agent.target_task = None
                    agent.g = agent.v_now  # Stay at delivery location
                    agent.elapsed = 0  # Reset elapsed time for priority calculation
            
            elif agent.is_targeting():
                # Agent is targeting pickup - check arrival
                if agent.v_now == agent.target_task.loc_pickup:
                    # Only assign if task is still available (not picked up by another agent)
                    # Use list membership here since we need to remove from list anyway
                    if agent.target_task in self.instance.tasks_unassigned:
                        # Move task from unassigned to assigned
                        self.instance.tasks_unassigned.remove(agent.target_task)
                        self.instance.tasks_assigned.append(agent.target_task)
                        self._assign_task(agent, agent.target_task)
                    else:
                        # Task was taken by another agent - clear target
                        # Will get new target in next assignment phase
                        agent.target_task = None
                        agent.g = agent.v_now
    
    def _func_pibt(self, agent: Agent, agent_list: list[Agent],
                   calling_agent: Agent | None = None) -> bool:
        """Core PIBT function for single agent planning with priority inheritance.
        
        Attempts to assign a collision-free next position for the given agent.
        If another agent occupies the desired position, recursively invokes PIBT
        for that agent (priority inheritance). Backtracks if no valid position
        is found.
        
        The algorithm prevents both vertex collisions (two agents at same position)
        and edge collisions (two agents swapping positions) through careful
        checking of occupied_next and calling_agent constraints.
        
        Args:
            agent: Agent requiring position planning.
            agent_list: List of Agent objects indexed by agent ID for O(1) access.
            calling_agent: Agent that invoked this function (prevents edge conflicts).
        
        Returns:
            True if successfully assigned a position to agent.v_next, False otherwise.
        """
        # Get candidate next positions (neighbors + stay) - use pre-computed neighbors
        neighbors = self.neighbors_cache.get(agent.v_now, [])
        candidates = neighbors + [agent.v_now]
        
        # Pre-compute distances and sort keys to avoid repeated calls
        candidates_with_keys = []
        for v in candidates:
            dist = self._get_path_dist(v, agent.g)
            # Prefer occupied positions when distance is equal (lower value = higher priority)
            # occupied=0 (prefer), empty=1 (avoid when occupied available)
            is_empty = 1 if self.occupied_now[v] == self.NIL else 0
            candidates_with_keys.append((v, dist, is_empty))
        
        # Sort by distance first, then occupancy
        candidates_with_keys.sort(key=lambda x: (x[1], x[2]))
        
        # Extract just the coordinates, maintaining sorted order
        candidates = [v for v, _, _ in candidates_with_keys]
        
        # Try each candidate
        for u in candidates:
            # Avoid vertex conflicts
            if self.occupied_next[u] != self.NIL:
                continue
            
            # Check who is currently at position u
            ak_id = self.occupied_now[u]
            
            # Avoid edge collision with calling agent
            # The calling agent called us to move, so we can't move to where it currently is
            if calling_agent is not None and u == calling_agent.v_now:
                continue
            
            # Avoid edge collision: if agent at u has already planned to move to our position
            if ak_id != self.NIL and ak_id != agent.id:
                ak = agent_list[ak_id]  # O(1) list access instead of dict lookup
                # If ak has already been planned and is moving to where we are now, that's a swap
                if ak.v_next is not None and ak.v_next == agent.v_now:
                    continue
            
            # Reserve next location
            self.occupied_next[u] = agent.id
            agent.v_next = u
            
            # Priority inheritance - recursively plan for agent at target position
            if ak_id != self.NIL and ak_id != agent.id:
                ak = agent_list[ak_id]  # O(1) list access
                # If agent at u hasn't planned yet, invoke PIBT for it
                if ak.v_next is None:
                    # Recursively plan for blocking agent, passing ourselves as calling_agent
                    if not self._func_pibt(ak, agent_list, agent):
                        # Failed to replan blocking agent, unreserve and try next candidate
                        self.occupied_next[u] = self.NIL
                        agent.v_next = None
                        continue
            
            # Successfully assigned position
            return True
        
        # Failed to find valid move: stay in place
        self.occupied_next[agent.v_now] = agent.id
        agent.v_next = agent.v_now
        return False
    
    def _get_path_dist(self, start: Coord, goal: Coord) -> int:
        """Get shortest path distance between two positions.
        
        Uses cached distance tables for efficient repeated queries. Each unique
        goal position gets its own DistTable computed once and reused.
        
        Args:
            start: Starting position (y, x).
            goal: Goal position (y, x).
        
        Returns:
            Shortest path distance in number of moves (Manhattan on grid).
        """
        # Use 2D array indexing instead of dict lookup (faster)
        if self.dist_tables_cache[goal] is None:
            self.dist_tables_cache[goal] = DistTable(self.grid, goal)
        return self.dist_tables_cache[goal].get(start)
    
    @staticmethod
    def _agent_priority_key(agent: Agent) -> tuple:
        """Compute priority key for agent sorting.
        
        Priority determines planning order in PIBT. Higher priority agents are
        planned first, giving them first choice of positions. This ensures agents
        with active tasks can make progress while free agents yield to them.
        
        Priority Hierarchy (highest to lowest):
            1. Assigned agents (carrying items to delivery) - highest priority
            2. Targeting agents (moving to pickup locations) - medium priority
            3. Free agents (no task) - lowest priority, should not block others
            4. Within each tier: Higher elapsed time (stuck longer = higher priority)
            5. Final tie-breaker: Random value for deterministic resolution
        
        Args:
            agent: Agent to compute priority for.
        
        Returns:
            Tuple (priority_tier, elapsed, tie_breaker) for sorting comparison.
            priority_tier: 2=assigned, 1=targeting, 0=free
        """
        if agent.is_assigned():
            priority_tier = 2  # Highest: carrying item to delivery
        elif agent.is_targeting():
            priority_tier = 1  # Medium: moving to pickup
        else:
            priority_tier = 0  # Lowest: free/idle, should yield to others
        
        return (priority_tier, agent.elapsed, agent.tie_breaker)
    
    def save_solution(self, output_file: str) -> None:
        """Save MAPD solution to file with task information.
        
        Outputs solution with agent trajectories and completed task information
        for proper MAPD visualization.
        
        Args:
            output_file: Path to output file.
            
        Example:
            >>> mapd = MAPD("instance.txt")
            >>> solution = mapd.run()
            >>> mapd.save_solution("output.txt")
        """
        save_mapd_solution(self.solution, output_file, self.agent_goals_history)
