# hw1_judge.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022
# Modified by dj4zo6u.6 on Mar 5, 2025

"""
This file contains the main application that judges the performance of
different search algorithms on various maze configurations.
"""

import time

from maze import Maze
from search import search

class Application:
    def __init__(self):
        pass

    def initialize(self, filename):
        self.maze = Maze(filename)
        self.gridDim = self.maze.getDimensions()

    def execute(self, filename, searchMethod):
        self.initialize(filename)

        if self.maze is None:
            print("No maze created")
            raise SystemExit

        t1 = time.time()
        path = search(self.maze, searchMethod)
        total_time = time.time() - t1  # time in seconds
        statesExplored = self.maze.getStatesExplored()
        print("total time:", total_time)
        if not self.maze.isValidPath(path):
            print("Invalid path")
            return -1
        if total_time > 1:
            print("Time limit exceeded")
            return -1
        else:
            return len(path)

if __name__ == "__main__":
    app = Application()
    # judge
    methods = [["bfs", "astar"], ["astar_corner"], ["astar_multi"], ["fast"]]
    directories = ["maps/single/", "maps/corner/", "maps/multi/", "maps/multi/"]
    files_ans = [
        [["tinyMaze.txt", 9], ["smallMaze.txt", 20], ["mediumMaze.txt", 69], ["bigMaze.txt", 211], ["openMaze.txt", 55]],
        [["tinyCorners.txt", 29], ["mediumCorners.txt", 107], ["bigCorners.txt", 163]],
        [["tinySearch.txt", 28], ["smallSearch.txt", 35], ["mediumSearch.txt", 169], ["mediumDottedMaze.txt", 75], ["greedySearch.txt", 17]],
        [["openSearch.txt", 0], ["oddSearch.txt", 0], ["mediumSearch_prev.txt", 0], ["bigSearch.txt", 0]]
    ]
    for pt in range(4):
        method = methods[pt]
        dir = directories[pt]
        files_ans_pairs = files_ans[pt]
        print(f"running part {pt+1}, method: {method}")
        p_correct, p_all = 0, 0
        for meth in method:
            for (file, ans) in files_ans_pairs:
                print(dir + file, " " * (40 - len(file)-len(dir)), end="")
                output = app.execute(dir + file, meth)
                if ans == 0 or output == ans:
                    p_correct += 1
                p_all += 1
        print(f"part {pt+1}: {p_correct} / {p_all}")
