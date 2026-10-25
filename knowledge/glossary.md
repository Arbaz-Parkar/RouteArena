# Glossary

## Objective

The number an optimization problem tries to minimize or maximize. In the route planner it is the total estimated tour time.

## Gradient

The direction and rate of the steepest increase of a smooth function. Moving in the opposite direction can reduce the objective.

## Tour

A closed sequence of stops that begins at the hub, visits every customer once, and returns to the hub.

## Exact algorithm

An algorithm that can prove it found the best answer for the defined input. Held-Karp is exact only within its supported small-stop limit.

## Heuristic

A practical search rule that can find a good answer without proving it is the best. Nearest neighbor, genetic search, and ant colony are heuristics.

## Monte Carlo

A method that samples many possible outcomes from a model to estimate a distribution, such as the chance a route misses a deadline.

## Percentile

A value below which a stated share of simulated outcomes falls. A 90th-percentile travel time is at least as large as 90 percent of the sampled times.

## Coefficient of variation

Standard deviation divided by mean. It expresses variation relative to the typical size of a quantity.

## Retrieval-augmented generation

A pattern that first retrieves relevant knowledge passages and then uses them to answer. RouteArena's current offline mode retrieves and quotes local notes without generating text through an external model.
