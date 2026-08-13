@wp.func
def in_circle(
        a : wp.vec2,
        b : wp.vec2,
        c : wp.vec2,
        d : wp.vec2) -> bool:
    """
    Returns True if point d is inside the circumcircle of triangle abc.

    Follows Shewchuk's "Predicates for Robust Geometric Computations" (1997) and uses adaptive precision arithmetic to ensure robustness.
    """



