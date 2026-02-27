import numpy as np
import math

EQUATIONS = [
    {
        "name": "x^3 + 2.28x^2 - 1.934x - 3.907",
        "f": lambda x: x**3 + 2.28 * x**2 - 1.934 * x - 3.907,
        "df": lambda x: 3 * x**2 + 4.56 * x - 1.934,
        "ddf": lambda x: 6 * x + 4.56,
    },
    {
        "name": "sin(x) - 0.5x",
        "f": lambda x: math.sin(x) - 0.5 * x,
        "df": lambda x: math.cos(x) - 0.5,
        "ddf": lambda x: -math.sin(x),
    },
    {
        "name": "x^2 - exp(x) + 2",
        "f": lambda x: x**2 - math.exp(x) + 2,
        "df": lambda x: 2 * x - math.exp(x),
        "ddf": lambda x: 2 - math.exp(x),
    },
]

SYSTEMS = [
    {
        "name": "Система №7 (2x - sin(y-0.5) = 1; y + cos(x) = 1.5)",
        "f_vec": lambda x, y: [2 * x - math.sin(y - 0.5) - 1, y + math.cos(x) - 1.5],
        "jacobian": lambda x, y: [[2, -math.cos(y - 0.5)], [-math.sin(x), 1]],
        "plot": [
            lambda x, y: 2 * x - np.sin(y - 0.5) - 1,
            lambda x, y: y + np.cos(x) - 1.5,
        ],
    },
    {
        "name": "Система №10 (sin(x+0.5) - y = 1; cos(y-2) + x = 0)",
        "f_vec": lambda x, y: [math.sin(x + 0.5) - y - 1, math.cos(y - 2) + x],
        "jacobian": lambda x, y: [[math.cos(x + 0.5), -1], [1, -math.sin(y - 2)]],
        "plot": [lambda x, y: np.sin(x + 0.5) - y - 1, lambda x, y: np.cos(y - 2) + x],
    },
]
