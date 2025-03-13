# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import heapq
from collections import deque

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    startPosition = maze.getStart()
    dots = maze.getObjectives()
    waitList = [startPosition]
    visitedPosition = set()
    parentPosition = {startPosition: None}
    path = []
    
    while True:
        currentPosition = waitList[0]
        waitList.pop(0)
        visitedPosition.add(currentPosition)

        #  Iterate its neighbors and check is valid or not 
        currentNeighbors = maze.getNeighbors(currentPosition[0], currentPosition[1])
        for neighbor in currentNeighbors:
            if neighbor not in visitedPosition:
                waitList.append(neighbor)
                # Record its parent to rebuild path
                parentPosition[neighbor] = currentPosition

        if set(dots).issubset(visitedPosition):
            while currentPosition:
                path.append(currentPosition)
                currentPosition = parentPosition[currentPosition]
            
            if maze.isValidPath(path[::-1]) == "Valid":
                return path[::-1] 
            else:
                path.clear()

    return path
    
def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def heuristic(position):
        sum = 0
        for dot in dots:
           sum += (dot[0] - position[0]) + (dot[1] - position[1]) 
        return sum
    
    startPosition = maze.getStart()
    dots = maze.getObjectives()
    waitMinheap = []
    parentPosition = {startPosition: None}
    visitedPosition = set()
    gcosts = {startPosition: 0}  # The cost from start position
    fcosts = {startPosition: heuristic(startPosition)} # Total cost = gcost + hcost
    path = []
    heapq.heappush(waitMinheap, (heuristic(startPosition), startPosition))

    while True:
        _, currentPosition = heapq.heappop(waitMinheap)
        currentNeighbors = maze.getNeighbors(currentPosition[0], currentPosition[1])
        visitedPosition.add(currentPosition)

        for neighbor in currentNeighbors:
            if (neighbor not in gcosts) or (gcosts[currentPosition] + 1 < gcosts[neighbor]):
                newgcost = gcosts[currentPosition] + 1
                newfcost = newgcost + heuristic(neighbor)

                gcosts[neighbor] = newgcost
                fcosts[neighbor] = newfcost
                heapq.heappush(waitMinheap,(newfcost, neighbor))
                # Record its parent to rebuild path
                parentPosition[neighbor] = currentPosition

        if set(dots).issubset(visitedPosition):
            while currentPosition:
                path.append(currentPosition)
                currentPosition = parentPosition[currentPosition]
            
            if maze.isValidPath(path[::-1]) == "Valid":
                return path[::-1] 
            else:
                path.clear()

    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    def heuristic(position):
        sum = 0
        for dot in dots:
           sum += (dot[0] - position[0]) + (dot[1] - position[1]) 
        return sum
    
    startPosition = maze.getStart()
    dots = maze.getObjectives()
    waitMinheap = []

    startState = (startPosition, frozenset())
    parentState = {startState: None}
    gcosts = {startState: 0} 
    fcosts = {startState: heuristic(startPosition)} # Total cost = gcost + hcost
    path = []
    heapq.heappush(waitMinheap, (heuristic(startPosition), 0, startState))

    while True:
        fcost, gcost, currentState = heapq.heappop(waitMinheap)
        currentPosition, currentVisited = currentState
        currentNeighbors = maze.getNeighbors(currentPosition[0], currentPosition[1])

        if len(currentVisited) == len(dots):
            while currentState is not None:
                path.append(currentState[0])
                currentState = parentState[currentState]
            
            if maze.isValidPath(path[::-1]) == "Valid":
                return path[::-1] 
            else:
                path.clear()

        for neighbor in currentNeighbors:
            newVisited = (currentVisited | {neighbor}) if (neighbor in dots) else currentVisited
            newState = (neighbor, frozenset(newVisited))

            if (newState not in gcosts) or (gcosts[currentState] + 1 < gcosts[newState]):
                newgcost = gcosts[currentState] + 1
                newfcost = newgcost + heuristic(neighbor)

                gcosts[newState] = newgcost
                fcosts[newState] = newfcost
                heapq.heappush(waitMinheap,(newfcost, newgcost, newState))
                # Record its parent to rebuild path
                parentState[newState] = currentState

    return path

def prim_mst_cost(objectives):
    """
    Computes the cost of the minimum spanning tree of the objectives using Prim's algorithm.

    @param objectives: The objectives to compute the minimum spanning tree on.

    @return cost: The cost of the minimum spanning tree.
    """
    if not objectives:
        return 0

    visited = set()
    visited.add(objectives[0])
    waitMinheap = []
    for neighbor in objectives[1:]:
        heapq.heappush(waitMinheap, (manhattan_distance(objectives[0], neighbor), neighbor))

    cost = 0
    while waitMinheap:
        minCost, current = heapq.heappop(waitMinheap)
        if current in visited:
            continue

        visited.add(current)
        cost += minCost

        for neighbor in objectives:
            if neighbor not in visited:
                heapq.heappush(waitMinheap, (manhattan_distance(current, neighbor), neighbor))

    return cost
    
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

from functools import lru_cache

@lru_cache(maxsize=None)
def cached_mst_cost(objectives):
    return prim_mst_cost(objectives)

def heuristic(position, objectives):
    if not objectives:
        return 0
    return cached_mst_cost(tuple(objectives)) + min(manhattan_distance(position, obj) for obj in objectives)

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start_position = maze.getStart()
    objectives = tuple(maze.getObjectives())
    start_state = (start_position, objectives)
    
    waitMinheap = []
    heapq.heappush(waitMinheap, (0, start_state))
    parentState = {start_state: None}
    gcosts = {start_state: 0}

    while waitMinheap:
        _, current_state = heapq.heappop(waitMinheap)
        current_position, current_objectives = current_state

        # Check if all objectives are reached
        if not current_objectives:
            path = []
            while current_state:
                path.append(current_state[0])
                current_state = parentState[current_state]
            return path[::-1]

        current_neighbors = maze.getNeighbors(current_position[0], current_position[1])
        for neighbor in current_neighbors:
            # If the neighbor is an objective, remove it from the objectives
            new_objectives = tuple(obj for obj in current_objectives if obj != neighbor)
            new_state = (neighbor, new_objectives)
            new_gcost = gcosts[current_state] + 1

            # If the new state is not visited or has a lower cost, update the cost and add it to the heap
            if new_state not in gcosts or new_gcost < gcosts[new_state]:
                gcosts[new_state] = new_gcost
                fcost = new_gcost + heuristic(neighbor, new_objectives)
                heapq.heappush(waitMinheap, (fcost, new_state))
                parentState[new_state] = current_state
                
    


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    def heuristic(state):
        sum = 0
        remainingDots = [dot for dot in dots if dot not in list(state[1])]
        for dot in remainingDots:
           sum += (dot[0] - state[0][0]) + (dot[1] - state[0][1]) 
        return sum
    
    startPosition = maze.getStart()
    dots = maze.getObjectives()
    waitMinheap = []

    # Initialize heuristic
    storeHeuristic = {}
    

    startState = (startPosition, frozenset())
    parentState = {startState: None}
    gcosts = {startState: 0} 
    fcosts = {startState: heuristic(startState)} # Total cost = gcost + hcost
    path = []
    heapq.heappush(waitMinheap, (heuristic(startState), 0, startState))

    while True:
        fcost, gcost, currentState = heapq.heappop(waitMinheap)
        currentPosition, currentVisited = currentState
        currentNeighbors = maze.getNeighbors(currentPosition[0], currentPosition[1])

        if len(currentVisited) == len(dots):
            while currentState is not None:
                path.append(currentState[0])
                currentState = parentState[currentState]
            
            if maze.isValidPath(path[::-1]) == "Valid":
                return path[::-1] 
            else:
                path.clear()

        for neighbor in currentNeighbors:
            newVisited = (currentVisited | {neighbor}) if (neighbor in dots) else currentVisited
            newState = (neighbor, frozenset(newVisited))

            if (newState not in gcosts) or (gcosts[currentState] + 1 < gcosts[newState]):
                newgcost = gcosts[currentState] + 1
                newfcost = newgcost + heuristic(newState)

                gcosts[newState] = newgcost
                fcosts[newState] = newfcost
                heapq.heappush(waitMinheap,(newfcost, newgcost, newState))
                # Record its parent to rebuild path
                parentState[newState] = currentState

    return path
