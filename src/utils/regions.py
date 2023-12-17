from dataclasses import dataclass


@dataclass
class BoundingBox:

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float


STR2BB = {
    "Togo": BoundingBox(
        min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625,
    ),
    "Nigeria": BoundingBox(min_lon=2.6917, max_lon=14.5772, min_lat=4.2406, max_lat=13.8659
    ), # https://gist.github.com/graydon/11198540 in W, S, E, N aprox. to 4 decimal
}
