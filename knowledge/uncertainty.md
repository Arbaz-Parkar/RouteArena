# Traffic uncertainty

## Monte Carlo model

Monte Carlo analysis repeats a route calculation with random travel-time factors. RouteArena draws one shared city-wide factor per simulated day, then draws separate independent factors for each leg. The shared factor makes delays on different legs move together, like a generally slow traffic day.

## Distribution and outputs

Both factors are lognormal and parameterized by arithmetic mean and coefficient of variation. Low, moderate, and heavy traffic use different assumed means and variation levels in `routearena.logistics.uncertainty`. The page reports the sample mean, spread, median, 90th and 95th percentiles, deadline risk, a histogram, and a convergence plot.

## Interval analysis

Interval analysis asks what happens if each leg stays between a chosen lower and upper multiplier. Adding the upper bounds gives the worst-case total under those limits. This is a set-based bound, not a probability or a forecast.

## Limitations

The traffic parameters are demonstration assumptions and are not measured traffic. The model uses a single shared daily factor and independent leg noise; it does not represent time of day, road-specific incidents, or changing traffic during a tour.
