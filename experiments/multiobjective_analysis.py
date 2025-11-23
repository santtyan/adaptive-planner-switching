"""
Multi-Objective Performance Analysis - Paper Enhancement
Path quality, energy efficiency, safety margins
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PathQualityAnalyzer:
    """Advanced path quality metrics for A1 papers"""
    
    def analyze_path_quality(self, path, environment):
        """Comprehensive path quality assessment"""
        if not path or len(path) < 2:
            return {'smoothness': 0, 'energy': float('inf'), 'safety': 0}
        
        # Path smoothness (curvature analysis)
        smoothness = self._compute_smoothness(path)
        
        # Energy consumption model
        energy = self._compute_energy_consumption(path)
        
        # Safety clearance from obstacles
        safety = self._compute_safety_margins(path, environment)
        
        # Length optimality
        optimality = self._compute_length_optimality(path)
        
        return {
            'smoothness': smoothness,
            'energy_consumption': energy,
            'safety_clearance': safety,
            'length_optimality': optimality,
            'composite_score': self._compute_composite_score(
                smoothness, energy, safety, optimality)
        }
    
    def _compute_smoothness(self, path):
        """Curvature-based smoothness metric"""
        if len(path) < 3:
            return 1.0
            
        angles = []
        for i in range(1, len(path)-1):
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if not angles:
            return 1.0
            
        # Smoothness = 1 - normalized angular variation
        angular_variation = np.std(angles)
        return max(0, 1 - angular_variation / np.pi)
    
    def _compute_energy_consumption(self, path):
        """Energy model based on distance and curvature"""
        total_energy = 0
        
        for i in range(len(path)-1):
            distance = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            
            # Base energy proportional to distance
            base_energy = distance * 0.1
            
            # Curvature penalty
            if i < len(path)-2:
                curvature_penalty = self._compute_curvature_penalty(path[i:i+3])
                total_energy += base_energy * (1 + curvature_penalty)
            else:
                total_energy += base_energy
                
        return total_energy
    
    def _compute_curvature_penalty(self, three_points):
        """Compute curvature penalty for energy consumption"""
        if len(three_points) != 3:
            return 0
            
        # Simple curvature approximation
        p1, p2, p3 = [np.array(p) for p in three_points]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
            
        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Higher angle = more curvature = more energy
        return (np.pi - angle) / np.pi * 0.5  # Max 50% penalty
    
    def _compute_safety_margins(self, path, environment):
        """Average clearance from obstacles"""
        if not hasattr(environment, 'grid'):
            return 0.5  # Default safety
            
        clearances = []
        for point in path:
            x, y = int(point[0]), int(point[1])
            
            # Check surrounding area for obstacles
            min_clearance = float('inf')
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < environment.grid_size and 
                        0 <= ny < environment.grid_size and
                        environment.grid[ny, nx] == 1):  # Obstacle
                        
                        clearance = np.sqrt(dx*dx + dy*dy)
                        min_clearance = min(min_clearance, clearance)
            
            if min_clearance != float('inf'):
                clearances.append(min_clearance)
            else:
                clearances.append(3.0)  # No nearby obstacles
        
        return np.mean(clearances) if clearances else 3.0
    
    def _compute_length_optimality(self, path):
        """Path length vs straight line distance"""
        if len(path) < 2:
            return 0
            
        # Actual path length
        actual_length = 0
        for i in range(len(path)-1):
            actual_length += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
        
        # Straight line distance
        straight_distance = np.linalg.norm(np.array(path[-1]) - np.array(path[0]))
        
        if straight_distance == 0:
            return 1.0
            
        # Optimality = straight_distance / actual_length
        optimality = straight_distance / actual_length
        return min(1.0, optimality)
    
    def _compute_composite_score(self, smoothness, energy, safety, optimality):
        """Weighted composite quality score"""
        # Normalize energy (inverse - lower is better)
        energy_normalized = 1.0 / (1.0 + energy)
        
        # Weighted combination
        weights = {'smoothness': 0.25, 'energy': 0.25, 'safety': 0.3, 'optimality': 0.2}
        
        composite = (weights['smoothness'] * smoothness + 
                    weights['energy'] * energy_normalized +
                    weights['safety'] * safety +
                    weights['optimality'] * optimality)
        
        return composite

if __name__ == "__main__":
    print("✅ PathQualityAnalyzer module ready")
