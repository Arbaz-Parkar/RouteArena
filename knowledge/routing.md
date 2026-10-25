# Delivery routing

## Problem

The routing page plans one rider's closed tour: start at hub index zero, visit every customer once, and return to the hub. It minimizes the sum of the supplied travel-time matrix. RouteArena currently makes that matrix from Haversine distances with an assumed detour factor and average speed.

## Methods and analogy

Nearest neighbor repeatedly picks the closest unvisited stop. Two-opt improves a tour by reconnecting two edges and reversing the route segment between them. Held-Karp dynamic programming checks enough partial tours to prove the best result for at most 12 customers. A genetic algorithm breeds customer-order permutations, while ant colony optimization reinforces edges that appear in shorter tours. The ant-colony analogy is ants leaving more scent on routes that led to food.

## Strengths and limits

Held-Karp gives an exact answer on small instances, but its work grows exponentially. Nearest neighbor is quick and easy to explain, though an early choice can lead to a poor finish. Genetic and ant-colony methods are randomized searches; a fixed seed makes a run reproducible but does not make a heuristic optimal.

## Where this appears

Delivery Planner compares the selected solvers on the same stops. The map draws straight connectors between locations; it does not compute turn-by-turn roads or live traffic routes.
