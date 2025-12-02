import argparse
import os

from pypibt import (
    PIBT,
    MAPD,
    get_grid,
    get_scenario,
    is_valid_mapf_solution,
    is_valid_mapd_solution,
    save_configs_for_visualizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIBT solver for MAPF and MAPD problems")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mapf", "mapd"],
        default="mapf",
        help="Problem type: MAPF (one-shot tasks) or MAPD (continuous pickup/delivery)"
    )
    
    # Common arguments
    parser.add_argument(
        "-m",
        "--map-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "maps", "random-32-32-10.map"
        ),
        help="Map file (.map format)"
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=str,
        help="Scenario file (.scen for MAPF, .txt for MAPD)"
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=200,
        help="Number of agents"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
        help="Output file path"
    )
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-timestep", type=int, default=1000, help="Maximum timesteps (default: 1000)")
    parser.add_argument("--max-comp-time", type=int, default=60000, help="Maximum computation time in milliseconds (default: 60000)")
    
    args = parser.parse_args()
    
    # Set default scenario files if not provided
    if args.scen_file is None:
        if args.mode == "mapf":
            args.scen_file = os.path.join(
                os.path.dirname(__file__), "assets", "mapf", "random-32-32-10-random-1.scen"
            )
        else:  # mapd
            args.scen_file = os.path.join(
                os.path.dirname(__file__), "assets", "mapd", "task-config.txt"
            )

    if args.mode == "mapf":
        # MAPF mode: one-shot path finding
        print("=== MAPF Mode ===")
        
        # Define problem instance
        grid = get_grid(args.map_file)
        starts, goals = get_scenario(args.scen_file, args.num_agents)
        
        print(f"Grid size: {grid.shape}")
        print(f"Agents: {len(starts)}")

        # Solve MAPF
        import time
        start_time = time.perf_counter()
        pibt = PIBT(grid, starts, goals, seed=args.seed)
        plan = pibt.run(max_timestep=args.max_timestep)
        comp_time = time.perf_counter() - start_time

        # Validation
        valid = is_valid_mapf_solution(grid, starts, goals, plan)
        print(f"Solved: {valid}")
        print(f"Timesteps: {len(plan) - 1}")
        print(f"Computation time: {comp_time:.2f}s")

        # Save result
        save_configs_for_visualizer(plan, args.output_file)
        print(f"Solution saved to: {args.output_file}")
    
    elif args.mode == "mapd":
        # MAPD mode: continuous pickup and delivery
        print("=== MAPD Mode ===")
        
        # Load and solve MAPD instance
        mapd = MAPD(args.scen_file, map_file=args.map_file, num_agents=args.num_agents, seed=args.seed)
        
        print(f"Grid size: {mapd.grid.shape}")
        print(f"Agents: {mapd.N}")
        print(f"Task frequency: {mapd.instance.task_frequency}")
        print(f"Total tasks: {mapd.instance.task_num}")
        
        # Run solver with command-line max_timestep and max_comp_time
        import time
        start_time = time.perf_counter()
        solution = mapd.run(max_timestep=args.max_timestep, max_comp_time=args.max_comp_time)
        comp_time = time.perf_counter() - start_time
        
        # Validation
        valid = is_valid_mapd_solution(
            mapd.grid, mapd.starts, solution, mapd.completed_tasks
        )
        
        # Results
        print(f"Solved: {valid}")
        print(f"Completed tasks: {len(mapd.completed_tasks)}/{mapd.instance.task_num}")
        print(f"Timesteps: {len(solution) - 1}")
        print(f"Computation time: {comp_time:.2f}s")
        
        # Save result
        mapd.save_solution(args.output_file)
        print(f"Solution saved to: {args.output_file}")
