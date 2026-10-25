# Project assumptions and FAQ

## Is this live delivery routing?

No. RouteArena is a local decision-support demonstration for one rider. It compares optimization methods and does not provide production directions, traffic feeds, multiple vehicles, or time windows.

## Where do the map points come from?

The default map region is Nerul, Navi Mumbai. Generated customer points are seeded synthetic examples near mapped roadside anchors. Uploaded CSV rows are checked for valid coordinates, duplicate points, stop limits, and numeric demand; the uploaded file is not edited.

## Are the route times measured?

No. The time estimate uses straight-line Haversine distance, a detour factor of 1.25, and an average speed of 25 kilometres per hour. These are assumptions. Map lines connect stops directly and are not road-following routes.

## Does an AI model choose the route?

No. Optimization algorithms compute every route and metric. The offline assistant searches local Markdown notes and saved run summaries, then gives a short answer with source citations. It does not call a language model or change solver results.

## Which planned features are not available yet?

Surrogate models, Bayesian tuning, the synthetic machine-learning lesson, and hosted language-model explanations are not part of the current app. The project should label these as future work until their code and evaluation are complete.

## Where are experiment records stored?

Each route or risk run is written as JSON under the git-ignored `runs/` folder. Records include settings, random seed, result summaries, timestamp, and library versions; they omit uploaded customer coordinates and names.
