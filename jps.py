import numpy as np
import PIL as pil 
from sortedcontainers import SortedList
import time

class GridMap(object):
    def __init__(self, image):
        # image is a path to an image
        if isinstance(image, str):
            self.map = np.where(np.asarray(pil.Image.open(image)) != 0, 1, 0)[:,:,0]
        else:
            self.map = np.asarray(image)
    
    def get_neighbors(self, x):
        neighbors = []
        
        for i in range(-1,2):
            for j in range(-1,2):
                # x is not a neighbor of itself
                if i == 0 and j == 0:
                    continue

                n = (x[0] + i, x[1] + j)
        
                if not self.is_obstacle(n):
                    neighbors.append(n)
        
        return neighbors
    
    def is_obstacle(self, x):
        if x[0] < 0 or x[0] >= self.map.shape[0] or x[1] < 0 or x[1] >= self.map.shape[1]:
            return True
        
        if self.map[x[0], x[1]] == 0:
            return True
        
        return False

# gets the euclidean distance between two points, a and b
def euclidean_distance(a, b):
    # convert to numpy arrays if necessary
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b)


def jps(grid, start, goal, heuristic_fn=euclidean_distance, max_iterations=10000):
"""
Performs Jump-Point Search on the grid map to find a path between the start and the goal. 
params:
    grid            - the grid to search on. 
                      requires two methods, is_obstacle(x: tuple or np.array) and get_neighbors(x : tuple or np.array)
    start           - the start location as a 2D tuple or np.array
    goal            - the goal location as a 2D tuple or a np.array
    heuristic_fn    - the heuristic function to use
    max_iterations  - the max number of iterations
"""

    def direction(a, b):
        d = np.asarray(b) - np.asarray(a)
        d = (d / max(np.abs(d[0]), np.abs(d[1]))).astype(int)
        return d

    def jps_is_natural_neighbor(x, parent, n):
        if (np.asarray(parent) == np.asarray(n)).all():
            return False
        
        if (np.asarray(x) == np.asarray(n)).all():
            return False
        
        # horizontal movement
        if parent[0] == x[0]:
            if x[0] == n[0]:
                return True
            else:
                return False
        
        # vertical movement
        if parent[1] == x[1]:
            if x[1] == n[1]:
                return True
            else:
                return False
        
        # diagonal movement and the neighbor is orthogonal to the parent
        if parent[0] == n[0] or parent[1] == n[1]:
            return False
        
        return True

    def jps_has_forced_neighbors(grid, x, p):
        d = direction(p,x)
        
        # to check for forced neighbors, we just need to check if certain locations have obstacles
        # the locations to check are determined by the direction of travel
        
        # moving vertically
        # need to check our left and right neighbors for obstacles
        if d[0] == 0:
            if grid.is_obstacle(x + np.asarray([1,0])) or grid.is_obstacle(x - np.asarray([1,0])):
                return True
            else:
                return False
        # moving horizontally
        # need to check our above and below neighbors for obstacles
        elif d[1] == 0:
            if grid.is_obstacle(x + np.asarray([0,1])) or grid.is_obstacle(x - np.asarray([0,1])):
                return True
            else:
                return False
        # moving diagonally
        # need to check the neighbors we share with the parent
        else:
            if grid.is_obstacle(np.asarray([x[0], p[1]])) or grid.is_obstacle(np.asarray([p[0], x[1]])):
                return True
            else:
                False

    def jps_jump(grid, x, d, s, g):
        n = np.asarray(x) + d
     
        # n is an obstacle
        if grid.is_obstacle(n):
            # print(f"Obstacle: {n}")
            return None
        
        # we've reached the goal
        if (n == np.asarray(g)).all():
            # print(f"Goal: {n}:")
            return n
        
        # n has forced neighbors
        if jps_has_forced_neighbors(grid, n, x):
            return n
        
        # if d is diagonal (i.e. if d = (1,1), (1, -1), (-1, 1) or (-1, -1))
        if d[0] != 0 and d[1] != 0:
            d_1 = np.array([d[0], 0])
            if jps_jump(grid, n, d_1, s, g) is not None:
                # print(f"Has orthogonal jump point: {n}")
                return n
            
            d_2 = np.array([0, d[1]])
            if jps_jump(grid, n, d_2, s, g) is not None:
                # print(f"Has orthogonal jump point: {n}")
                return n
        
        return jps_jump(grid, n, d, s, g)
        
    def jps_get_pruned_neighbors(grid, x, d):
        pruned = []
        x = np.asarray(x)
        
        # we always want to continue going in the direction of travel
        if not grid.is_obstacle(x + d):
            pruned.append(x + d)
        
        # by taking account the direction of travel, we can handle all the rotationally symmetric cases in one go
       
        # we are moving horizontally, so we are checking if o is an obstacle and adding nearby n if it is
        # _ o n
        # p x _
        # _ o n
        if d[0] == 0:
            o1 = x + np.asarray([1,0])
            if grid.is_obstacle(o1) and not grid.is_obstacle(o1 + d):
                pruned.append(o1 + d)
            
            o2 = x - np.asarray([1,0])
            if grid.is_obstacle(o2) and not grid.is_obstacle(o2 + d):
                pruned.append(o2 + d)
        # we are moving vertically, so we are checking if o is an obstacle and adding the nearby n if it is
        # n _ n
        # o x o
        # _ p _
        elif d[1] == 0:
            o1 = x + np.asarray([0,1])
            if grid.is_obstacle(o1) and not grid.is_obstacle(o1 + d):
                pruned.append(o1 + d)
            
            o2 = x - np.asarray([0,1])
            if grid.is_obstacle(o2) and not grid.is_obstacle(o2 + d):
                pruned.append(o2 + d)
        # we are moving diagonally, so we are checking if o is an obstacle and adding the nearby n if it is
        # the numbers are always added unless they are obstacles
        # n 1 _
        # o x 2
        # p o n
        else:
            d0 = np.asarray([d[0],0])
            d1 = np.asarray([0,d[1]])
            n1 = x + d0
            n2 = x + d1
            
            if not grid.is_obstacle(n1):
                pruned.append(n1)
            
            if not grid.is_obstacle(n2):
                pruned.append(n2)
                
            o1 = x - np.asarray(d0)
            n3 = x - d0 + d1
            
            if grid.is_obstacle(o1) and not grid.is_obstacle(n3):
                 pruned.append(n3)
            
            o2 = x - np.asarray(d1)
            n4 = x - d1 + d0
            
            if grid.is_obstacle(o2) and not grid.is_obstacle(n4):
                pruned.append(n4)
        return pruned

    def jps_get_successors(grid, x, parent, start, goal):
        if parent is None:
            return grid.get_neighbors(x)
        
        x = np.asarray(x)
        parent = np.asarray(parent)
        
        p_d = direction(parent, x)
        
        successors = []
        neighbors = jps_get_pruned_neighbors(grid, x, p_d)
        
        # jump!
        for n in neighbors:
            d = direction(x,n)
            n = jps_jump(grid, x, d, start, goal)
            
            if n is not None:
                successors.append(n)
                if (n == np.asarray(goal)).all():
                    return [n]
        
        return successors
    stime = time.time()
    
    # Cost function: distance from start (q[0]) + heuristic cost (q[1])
    cost_fn = lambda q: q[0] + q[1]
    
    # Q is a "sorted list" of tuples:
    #  (dist from start, heuristic, list of states in current 'branch')
    Q = SortedList([(0, heuristic_fn(start, goal), [start])], key=cost_fn)
    visited = set()
    for ind in range(max_iterations):
        if len(Q) == 0:
            break
            
        #print(f"Iteration: {ind}")
        # This gets N with the smallest value of cost_fn and removes it from Q
        N = Q.pop(0)
        
        # Split up N (for convenience)
        distN, hN, statesN = N
        
        # Check if the goal has been reached
        if (np.asarray(statesN[-1]) == np.asarray(goal)).all():
            return {
                'succeeded': True,
                'path': statesN,
                'num_iterations': ind,
                'path_len': len(statesN),
                'num_visited': len(visited),
                'visited_list': list(visited),
                'time': time.time() - stime,
            }
        if tuple(statesN[-1]) in visited:
            continue
        else:
            visited.add(tuple(statesN[-1]))
        
        # Then add new paths to Q from N (and its children)
        for c in successor_fn(grid, statesN[-1], statesN[-2] if len(statesN) >= 2 else None, start, goal):
            c = tuple(c)
            if c in visited:
                continue
            Q.add((distN+euclidean_distance(statesN[-1], c), heuristic_fn(c, goal), statesN + [c]))
        
    return {'succeeded': False}
