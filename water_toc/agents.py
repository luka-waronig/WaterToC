from mesa import Agent
from enum import Enum
from pydantic import BaseModel
import random
import numpy as np

"""
Defines the agent classes for the WaterToC agent-based model.

This module includes:
- Strategy (Enum): Defines the possible actions for agents (COOPERATE, DEFECT).
- BaseAgent (Class): A parent class with shared logic for all agents, including
  game theory calculations based on environmental state (Weitz et al., 2016).
- Human (Class): An agent representing human actors with specific consumption rates.
- AI (Class): An agent representing AI actors with their own consumption rates.

Agents decide their strategy based on local water availability, with a configurable
rate of deviation from their optimal choice.
"""

#strategy enum
class Strategy(str, Enum):
    COOPERATE = "C"
    DEFECT = "D"

#base agent class for humans and ai
class BaseAgent(Agent):
    def __init__(self, unique_id, model, pos=None):
        super().__init__(unique_id, model)
        self.pos = pos

    #shorthand for accessing strategies
    @property
    def COOPERATE(self):
        return Strategy.COOPERATE

    @property
    def DEFECT(self):
        return Strategy.DEFECT

    def set_game(self,R,S,T,P):
        """set the game-theoretic payoff matrix."""
        self.game = np.array([[R,S],[T,P]])
        return True

    def weitz_matrix_env(self, n):
        """
        Creates an environment-dependent payoff matrix based on Weitz et al. 2016.
        A(n) = (1-n) * [[T, P], [R, S]] + n * [[R, S], [T, P]]

        args:
            n: environmental state (0 = degraded, 1 = pristine)
        
        returns:
            env_matrix: 2x2 numpy array for the environment-dependent payoff matrix.
        """
        #read payoffs from the game matrix
        R = self.game[0][0]  #reward for mutual cooperation
        S = self.game[0][1]  #sucker's payoff
        T = self.game[1][0]  #temptation to defect
        P = self.game[1][1]  #punishment for mutual defection

        #Weitz formula: A(n) = (1-n) * degraded_matrix + n * pristine_matrix
        degraded_matrix = np.array([[T, P], [R, S]])
        pristine_matrix = np.array([[R, S], [T, P]])

        #interpolate between degraded and pristine states
        env_matrix = (1 - n) * degraded_matrix + n * pristine_matrix

        return env_matrix

    def choose_best_action(self, env_game):
        """
        Chooses the best action (cooperate or defect) by comparing the sum of payoffs
        in the environment-dependent game matrix.

        returns:
            strategy (Strategy): the optimal strategy.
        """
        row_sums = env_game.sum(axis=1)
        best_row_index = int(np.argmax(row_sums))
        strategy = self.COOPERATE if best_row_index == 0 else self.DEFECT
        return strategy

    def choose_action_with_deviation(self, env_game, deviation_rate):
        """
        Chooses an action, allowing for a random deviation from the optimal strategy.
        
        args:
            env_game: the environment-dependent payoff matrix.
            deviation_rate: probability of deviating from the optimal strategy.
        
        returns:
            strategy (Strategy): the selected strategy.
        """
        #determine the optimal strategy
        optimal_strategy = self.choose_best_action(env_game)

        #check for a random deviation
        if self.model.random.random() < deviation_rate:
            #deviate by choosing the opposite strategy
            return self.DEFECT if optimal_strategy == self.COOPERATE else self.COOPERATE
        else:
            #otherwise, follow the optimal strategy
            return optimal_strategy

    def get_neighbors(self, radius=1, include_center=False):
        """Get neighboring agents."""
        if self.pos is None:
            return []
        return self.model.grid.get_neighbors(self.pos, moore=True, include_center=include_center, radius=radius)


class Human(BaseAgent):
    def __init__(self, unique_id, model, strategy: Strategy):
        super().__init__(unique_id, model, pos=None)
        self.strategy = strategy
        self.human_C_allocation = model.human_C_allocation
        self.human_D_allocation = model.human_D_allocation

        #store action for simultaneous execution
        self.planned_action = None
        self.target_pos = None

    def step(self):
        """
        During each step, the agent identifies a nearby water source and plans its action
        (cooperate or defect) based on the water level.
        """
        water_positions = self.model.get_water_positions_near(self.pos)

        if not water_positions:
            return  

        #pick a random, nearby water cell
        self.target_pos = self.model.random.choice(water_positions)

        #calculate the environmental state 'n' for the target cell
        water_level = self.model.get_water_at(self.target_pos)
        max_capacity = self.model.water_capacity[self.target_pos[0], self.target_pos[1]]
        n = water_level / max_capacity if max_capacity > 0 else 0

        #choose strategy based on the environment, with a chance to deviate
        env_game = self.weitz_matrix_env(n)
        self.strategy = self.choose_action_with_deviation(env_game, self.model.deviation_rate)
        self.planned_action = self.strategy

    def advance(self):
        """Execute the planned action to consume water."""
        if self.planned_action and self.target_pos:
            if self.planned_action == self.COOPERATE:
                self.model.consume_water_at(self.target_pos, self.human_C_allocation)
            else:
                self.model.consume_water_at(self.target_pos, self.human_D_allocation)

        #reset planned action for the next tick
        self.planned_action = None
        self.target_pos = None

class AI(BaseAgent):
    def __init__(self, unique_id, model, strategy: Strategy):
        super().__init__(unique_id, model, pos=None)
        self.strategy = strategy
        self.ai_C_allocation = model.ai_C_allocation
        self.ai_D_allocation = model.ai_D_allocation

        #store action for simultaneous execution
        self.planned_action = None
        self.target_pos = None

    def step(self):
        """
        During each step, the agent identifies a nearby water source and plans its action
        (cooperate or defect) based on the water level.
        """
        water_positions = self.model.get_water_positions_near(self.pos)

        if not water_positions:
            return  # no water nearby

        #pick a random, nearby water cell
        self.target_pos = self.model.random.choice(water_positions)

        #calculate the environmental state 'n' for the target cell
        water_level = self.model.get_water_at(self.target_pos)
        max_capacity = self.model.water_capacity[self.target_pos[0], self.target_pos[1]]
        n = water_level / max_capacity if max_capacity > 0 else 0

        #choose strategy based on the environment, with a chance to deviate
        env_game = self.weitz_matrix_env(n)
        self.strategy = self.choose_action_with_deviation(env_game, self.model.deviation_rate)
        self.planned_action = self.strategy

    def advance(self):
        """Execute the planned action to consume water."""
        if self.planned_action and self.target_pos:
            if self.planned_action == self.COOPERATE:
                self.model.consume_water_at(self.target_pos, self.ai_C_allocation)
            else:
                self.model.consume_water_at(self.target_pos, self.ai_D_allocation)
                
        #reset planned action for the next tick
        self.planned_action = None
        self.target_pos = None