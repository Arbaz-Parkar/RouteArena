# Optimization methods

## What an optimizer does

An optimizer searches for a decision that makes an objective smaller or larger. In RouteArena the objective is written as a function, and the decision is a vector such as a hub's east and north position. An optimizer returns its final decision and a path of the guesses it tried.

## First-order methods

Gradient descent uses the slope of the objective to step downhill. It is like walking down a foggy hill by feeling which way the ground tilts. Adagrad, RMSProp, Adadelta, and Adam adjust step sizes using recent gradient information. These methods work well on many smooth problems, but a poor step size or a difficult landscape can make them slow or unstable.

## Second-order methods

Newton's method uses both slope and curvature to estimate a nearby minimum. Secant search estimates curvature from successive points, while BFGS builds a curvature approximation from gradient changes. Curvature can make progress fast near a good solution, but Newton steps can behave badly when curvature is flat or points away from a minimum.

## One-dimensional searches

Golden-section, Fibonacci, and quadratic-fit searches narrow an interval that contains a minimum. RouteArena uses golden-section search as a line search for gradient descent. These methods assume the chosen interval is useful and the function is well behaved there.

## Where this appears

The Optimizer Arena draws paths on quadratic and Rosenbrock surfaces. Hub Placement also uses first- and second-order methods on a smooth weighted-distance objective.
