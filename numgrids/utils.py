
def as_array_like(x):
    if hasattr(x, "__len__"):
        return x
    return (x,)

