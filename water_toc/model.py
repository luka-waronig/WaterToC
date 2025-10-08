from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import random
from mesa.time import SimultaneousActivation
from water_toc.agents import AI, Human, Strategy
import os
import json
from datetime import datetime


class WaterToC(Model):
    """
    An agent-based model exploring cooperation dynamics in a shared resource environment.
    The model was inspired by Weitz et al. (2016) paper titled "An oscillating tragedy of 
    the commons in replicator dynamics with game-environment feedback".
    """
    def __init__(self,
                 height=20,
                 width=20,
                 initial_humans=50,
                 initial_ai=50,
                 human_C_allocation=0.1,
                 human_D_allocation=0.15,
                 ai_C_allocation=2.0,
                 ai_D_allocation=3.0,
                 max_water_capacity=20,
                 water_cell_density=0.3,
                 theta=3.0,
                 deviation_rate=0.1,
                 save_snapshots=True,
                snapshot_interval=1,
                snapshot_dir="snapshots",
                 seed=None):
        """Initializes the Water Tragedy of Commons model."""
        super().__init__(seed=seed)

        #store all parameters passed from the server
        self.height = height
        self.width = width
        self.initial_humans = initial_humans
        self.initial_ai = initial_ai
        self.human_C_allocation = human_C_allocation
        self.human_D_allocation = human_D_allocation
        self.ai_C_allocation = ai_C_allocation
        self.ai_D_allocation = ai_D_allocation
        self.max_water_capacity = max_water_capacity
        self.water_cell_density = water_cell_density
        self.theta = theta
        self.deviation_rate = deviation_rate

        #initialize grid, agent list, and model running state
        self.grid = MultiGrid(self.width, self.height, True)
        self.agents = []
        self.running = True
        self.schedule = SimultaneousActivation(self)   #create a scheduler

        self._create_water_environment()
        self._create_agents()       
    
        self.save_snapshots = save_snapshots
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.step_count = 0
        #set up the data collector
        reporters = {
            "Total_Water": self._get_total_water,
            "Total_Water_Capacity": self._get_total_water_capacity,
            "Environment_State": self._get_environment_state,
            "Human_Count": lambda m: self._count_agents(agent_type=Human),
            "AI_Count": lambda m: self._count_agents(agent_type=AI),
            "Cooperators": self._count_cooperators,
            "Defectors": self._count_defectors,
            "Coop_Fraction": self._get_coop_fraction,
            "Human_Coop_Fraction": self._get_human_coop_fraction,
            "AI_Coop_Fraction": self._get_ai_coop_fraction,
            "theta": lambda m: m.theta,
            "Avg_Water_Per_Cell": self._get_avg_water_per_cell,
            "Local_Coop_Variance": lambda m: np.var(m._get_local_cooperation_map()),
            "Coop_Map_Flat": self._get_local_cooperation_map_flat,
            "Agent_Pos_Strats": self._get_agent_pos_strategies_list,
            "HasWater_Flat": lambda m: m.has_water.flatten().astype(int).tolist()
}

        #sample k water cells for local water level reporting
        water_positions = list(zip(*np.where(self.has_water)))
        k = 3
        if len(water_positions) >= k:
            self.sample_cells = self.random.sample(water_positions, k=k)
            for i, (x, y) in enumerate(self.sample_cells):
                reporters[f"n_cell_{i}"] = (lambda m, x=x, y=y:
                                             m.water_levels[x, y] / m.water_capacity[x, y] if m.water_capacity[x, y] > 0 else 0)

        self.datacollector = DataCollector(model_reporters=reporters)
        self.datacollector.collect(self)

    def _create_water_environment(self):
        """Initializes the grid with water resources."""
        self.water_levels = np.zeros((self.width, self.height))
        self.water_capacity = np.zeros((self.width, self.height))
        self.replenishment_rates = np.zeros((self.width, self.height))
        self.has_water = np.zeros((self.width, self.height), dtype=bool)

        for x in range(self.width):
            for y in range(self.height):
                #each cell has a chance to contain water based on density
                if self.random.random() < self.water_cell_density:
                    self.has_water[x, y] = True
                    self.water_capacity[x, y] = self.max_water_capacity
                    self.water_levels[x, y] = self.max_water_capacity #start at full capacity
                    self.replenishment_rates[x, y] = self.random.uniform(0.1, 0.5)

    def _create_agents(self):
        """Creates and places Human and AI agents."""
        #create human agents
        for _ in range(self.initial_humans):
            strategy = self.random.choice([Strategy.COOPERATE, Strategy.DEFECT])
            agent = Human(self.next_id(), self, strategy)
            agent.set_game(R=3, S=0, T=5, P=1)
            #pass model-level allocation parameters to each agent
            agent.C_allocation = self.human_C_allocation
            agent.D_allocation = self.human_D_allocation
            pos = self._get_random_empty_cell()
            if pos:
                self.grid.place_agent(agent, pos)
                agent.pos = pos
                self.agents.append(agent)
                self.schedule.add(agent)   
        #create AI agents
        for _ in range(self.initial_ai):
            strategy = self.random.choice([Strategy.COOPERATE, Strategy.DEFECT])
            agent = AI(self.next_id(), self, strategy)
            agent.set_game(R=3, S=0, T=5, P=1)
            #pass model-level allocation parameters to each agent
            agent.C_allocation = self.ai_C_allocation
            agent.D_allocation = self.ai_D_allocation
            pos = self._get_random_empty_cell()
            if pos:
                self.grid.place_agent(agent, pos)
                agent.pos = pos
                self.agents.append(agent)
                self.schedule.add(agent)   

    def _get_random_empty_cell(self):
        """Finds a random grid cell with space for another agent."""
        attempts = 0
        while attempts < 100:
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            pos = (x, y)
            #allow up to 3 agents per cell
            if len(self.grid.get_cell_list_contents([pos])) < 3:
                return pos
            attempts += 1
        return (0, 0) #default fallback

    def replenish_water(self):
        """Updates water levels in all cells based on local cooperation."""
        for x in range(self.width):
            for y in range(self.height):
                if not self.has_water[x, y]:
                    continue
                
                #get all agents within the feedback radius
                neighbors = self.grid.get_neighbors((x, y), moore=True, radius=3, include_center=True)
                local_cooperators = sum(1 for agent in neighbors if hasattr(agent, 'strategy') and agent.strategy == Strategy.COOPERATE)
                local_defectors = sum(1 for agent in neighbors if hasattr(agent, 'strategy') and agent.strategy == Strategy.DEFECT)
                
                local_total = local_cooperators + local_defectors
                if local_total == 0:
                    effective_replenishment = self.replenishment_rates[x, y]
                else:
                    local_coop_fraction = local_cooperators / local_total
                    #calculate feedback strength based on Weitz et al.
                    local_feedback_strength = self.theta * local_coop_fraction - 1.0
                    #factor based on logistic growth (highest impact at 50% saturation)
                    current_n = self.water_levels[x, y] / self.water_capacity[x, y] if self.water_capacity[x, y] > 0 else 0
                    env_capacity_factor = current_n * (1 - current_n)
                    #combine terms to get the final feedback multiplier
                    # Never below baseline; keep the positive side unchanged
                    local_feedback_term = np.clip(1.0 + 0.5 * env_capacity_factor * local_feedback_strength, 1.0, 3.0)
                    effective_replenishment = self.replenishment_rates[x, y] * local_feedback_term

                #apply replenishment, ensuring it doesn't exceed capacity
                self.water_levels[x, y] = min(self.water_capacity[x, y], self.water_levels[x, y] + effective_replenishment)

    def get_water_at(self, pos):
        """Gets the water level at a specific position."""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            return self.water_levels[x, y]
        return 0

    def consume_water_at(self, pos, amount):
        """Consumes water at a position and returns the amount actually consumed."""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            actual = min(amount, self.water_levels[x, y])
            self.water_levels[x, y] -= actual
            return actual
        return 0
    
    def add_water_at(self, pos, amount):
        """Adds water at a position, up to its capacity."""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            self.water_levels[x, y] = min(self.water_capacity[x, y], self.water_levels[x, y] + amount)

    def get_water_positions_near(self, pos, radius=1):
        """Gets a list of positions with water near a given location."""
        neighbors = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=radius)
        return [neighbor for neighbor in neighbors if self.get_water_at(neighbor) > 0]

    def _get_local_cooperation_map(self):
        """Creates a 2D array representing the local cooperation fraction at each cell."""
        coop_map = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                neighbors = self.grid.get_neighbors((x, y), moore=True, radius=2, include_center=True)
                agents_in_neighborhood = [agent for agent in neighbors if hasattr(agent, 'strategy')]
                local_total = len(agents_in_neighborhood)
                if local_total > 0:
                    local_cooperators = sum(1 for agent in agents_in_neighborhood if agent.strategy == Strategy.COOPERATE)
                    coop_map[x, y] = local_cooperators / local_total
                else:
                    coop_map[x, y] = 0.5 
        return coop_map
    
    def _save_spatial_snapshot(self):
        base = os.path.join(self.snapshot_dir, "snapshots.jsonl")

        coop_map = self._get_local_cooperation_map().tolist()
        agents = self._get_agent_pos_strategies_table()

        row = {
            "step": self.step_count,
            "coop_map": coop_map,
            "agents": agents,
            "water_levels": self.water_levels.tolist(),
            "water_capacity": self.water_capacity.tolist(),
            "has_water": self.has_water.tolist(),
        }

        with open(base, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")



    def step(self):
    # environment updates first
        self.replenish_water()
    # scheduler runs both phases (step and advance) for all agents
        self.schedule.step()
    # bookkeeping
        self.step_count += 1
    # records data
        self.datacollector.collect(self)
    # optionally persist spatial snapshots
        if self.save_snapshots and (self.step_count % self.snapshot_interval == 0):
            self._save_spatial_snapshot()


###Reporter methods for data collection

    def _get_total_water(self):
        """Calculates the sum of water across all cells."""
        return np.sum(self.water_levels[self.has_water])

    def _get_total_water_capacity(self):
        """Calculates the total possible water capacity of the system."""
        return np.sum(self.water_capacity[self.has_water])

    def _get_environment_state(self):
        """Calculates the overall health of the environment as a fraction."""
        total_capacity = self._get_total_water_capacity()
        return self._get_total_water() / total_capacity if total_capacity > 0 else 0

    def _get_avg_water_per_cell(self):
        """Calculates the average water level per water-containing cell."""
        water_cells_count = np.sum(self.has_water)
        return self._get_total_water() / water_cells_count if water_cells_count > 0 else 0

    def _count_agents(self, agent_type=None, strategy=None):
        """Helper method to count agents with optional filters for type and strategy."""
        return sum(1 for agent in self.agents if (agent_type is None or isinstance(agent, agent_type)) and \
                   (strategy is None or (hasattr(agent, 'strategy') and agent.strategy == strategy)))

    def _count_cooperators(self):
        """Counts the total number of cooperating agents."""
        return self._count_agents(strategy=Strategy.COOPERATE)

    def _count_defectors(self):
        """Counts the total number of defecting agents."""
        return self._count_agents(strategy=Strategy.DEFECT)

    def _get_coop_fraction(self):
        """Calculates the overall fraction of cooperators in the population."""
        total = len(self.agents)
        return self._count_cooperators() / total if total > 0 else 0

    def _get_human_coop_fraction(self):
        """Calculates the fraction of Human agents who are cooperating."""
        total_humans = self._count_agents(agent_type=Human)
        if total_humans == 0: return 0
        human_cooperators = self._count_agents(agent_type=Human, strategy=Strategy.COOPERATE)
        return human_cooperators / total_humans

    def _get_ai_coop_fraction(self):
        """Calculates the fraction of AI agents who are cooperating."""
        total_ai = self._count_agents(agent_type=AI)
        if total_ai == 0: return 0
        ai_cooperators = self._count_agents(agent_type=AI, strategy=Strategy.COOPERATE)
        return ai_cooperators / total_ai
    
    def _get_local_cooperation_map_flat(self):
    
        return self._get_local_cooperation_map().flatten().tolist()

    def _get_agent_pos_strategies_list(self):
    
        out = []
        for a in self.agents:
            if getattr(a, "pos", None) is None:
                continue
            kind = type(a).__name__
            sx, sy = a.pos
            strat = getattr(a, "strategy", None)
            strat = getattr(strat, "value", None)
            out.append((int(a.unique_id), kind, int(sx), int(sy), strat))
        return out

    def _get_agent_pos_strategies_table(self):
    
        rows = []
        for a in self.agents:
            if getattr(a, "pos", None) is None:
                continue
            sx, sy = a.pos
            rows.append({
                "id": int(a.unique_id),
                "type": type(a).__name__,
                "x": int(sx),
                "y": int(sy),
                "strategy": getattr(getattr(a, "strategy", None), "value", None),
            })
        return rows

