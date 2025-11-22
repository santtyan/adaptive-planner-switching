# src/planners/rrt_star_impl.py - Implementação própria RRT*
import numpy as np
import math
from typing import List, Tuple, Optional
import sys
sys.path.append('src')
from environment import SimpleEnvironment

class Node:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['Node'] = None
        self.cost = 0.0
        
    def distance_to(self, other: 'Node') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class RRTStarPlanner:
    '''Implementação própria RRT* - sem dependência OMPL'''
    
    def __init__(self, env: SimpleEnvironment):
        self.env = env
        self.step_size = 5.0
        self.search_radius = 10.0
        self.max_iterations = 5000
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             time_limit: float = 2.0) -> Tuple[bool, float, List[Tuple[float, float]]]:
        '''Plan path using RRT*'''
        import time
        t0 = time.time()
        
        # Initialize tree
        start_node = Node(start[0], start[1])
        goal_node = Node(goal[0], goal[1])
        
        nodes = [start_node]
        
        for i in range(self.max_iterations):
            # Check time limit
            if time.time() - t0 > time_limit:
                break
                
            # Sample random point
            rand_node = self._sample_random()
            
            # Find nearest node
            nearest_node = self._find_nearest(nodes, rand_node)
            
            # Steer towards sample
            new_node = self._steer(nearest_node, rand_node)
            
            # Check collision
            if self._is_collision_free(nearest_node, new_node):
                # Find nearby nodes for rewiring
                near_nodes = self._find_near_nodes(nodes, new_node)
                
                # Choose best parent
                best_parent = self._choose_parent(near_nodes, new_node)
                if best_parent:
                    new_node.parent = best_parent
                    new_node.cost = best_parent.cost + best_parent.distance_to(new_node)
                    
                nodes.append(new_node)
                
                # Rewire tree
                self._rewire(near_nodes, new_node)
                
                # Check if goal reached
                if new_node.distance_to(goal_node) < self.step_size:
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + new_node.distance_to(goal_node)
                    
                    # Extract path
                    path = self._extract_path(goal_node)
                    planning_time = (time.time() - t0) * 1000
                    return True, planning_time, path
        
        # No path found
        planning_time = (time.time() - t0) * 1000
        return False, planning_time, []
    
    def _sample_random(self) -> Node:
        '''Sample random point in space'''
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        return Node(x, y)
    
    def _find_nearest(self, nodes: List[Node], target: Node) -> Node:
        '''Find nearest node to target'''
        min_dist = float('inf')
        nearest = nodes[0]
        for node in nodes:
            dist = node.distance_to(target)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest
    
    def _steer(self, from_node: Node, to_node: Node) -> Node:
        '''Steer from from_node towards to_node with step size limit'''
        dist = from_node.distance_to(to_node)
        if dist <= self.step_size:
            return Node(to_node.x, to_node.y)
        
        # Limit step size
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * math.cos(theta)
        new_y = from_node.y + self.step_size * math.sin(theta)
        
        return Node(new_x, new_y)
    
    def _is_collision_free(self, node1: Node, node2: Node) -> bool:
        '''Check if path between nodes is collision-free'''
        # Simple line collision check
        steps = int(node1.distance_to(node2))
        if steps == 0:
            return self.env.is_valid(node2.x, node2.y)
            
        for i in range(steps + 1):
            t = i / steps
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            if not self.env.is_valid(x, y):
                return False
        return True
    
    def _find_near_nodes(self, nodes: List[Node], new_node: Node) -> List[Node]:
        '''Find nodes within search radius'''
        near_nodes = []
        for node in nodes:
            if node.distance_to(new_node) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes
    
    def _choose_parent(self, near_nodes: List[Node], new_node: Node) -> Optional[Node]:
        '''Choose best parent from near nodes'''
        if not near_nodes:
            return None
            
        best_parent = None
        min_cost = float('inf')
        
        for node in near_nodes:
            if self._is_collision_free(node, new_node):
                cost = node.cost + node.distance_to(new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node
                    
        return best_parent
    
    def _rewire(self, near_nodes: List[Node], new_node: Node):
        '''Rewire tree to minimize costs'''
        for node in near_nodes:
            new_cost = new_node.cost + new_node.distance_to(node)
            if (new_cost < node.cost and 
                self._is_collision_free(new_node, node)):
                node.parent = new_node
                node.cost = new_cost
    
    def _extract_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        '''Extract path from goal to start'''
        path = []
        current = goal_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]  # Reverse to get start->goal

if __name__ == '__main__':
    # Test implementation
    env = SimpleEnvironment(obstacle_density=0.2, seed=42)
    planner = RRTStarPlanner(env)
    
    success, time_ms, path = planner.plan((10, 10), (90, 90))
    print(f'Success: {success}, Time: {time_ms:.1f}ms, Path length: {len(path)}')
