# Hub placement

## Problem and objective

Hub placement chooses one point to serve a set of customer locations. RouteArena minimizes the demand-weighted sum of smoothed straight-line distances from the hub to customers. Smoothing replaces distance d with sqrt(d squared plus epsilon squared), which avoids an undefined gradient exactly at a customer.

## Everyday analogy

Imagine choosing a meeting point for a group, where people who represent more customers count more. A good point is near the heavier clusters, but the best point is not always the geographic center.

## Coordinates and methods

Latitude and longitude are projected into a local flat coordinate system near Nerul before gradient-based optimization. This makes the derivatives easier to scale and interpret. The page overlays optimizer paths on a contour plot of the objective.

## When it works and when it fails

This is useful for comparing continuous optimizers on a small, smooth example. Straight-line distance ignores roads, barriers, and delivery access, so the result is a teaching estimate rather than a recommended store address. A different projection is needed for locations far from the reference area.
