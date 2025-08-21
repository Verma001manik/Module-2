"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterator, List

#
# Implementation of a prelude of elementary functions.



# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def log(x: float) -> float:
    """Computes the natural logarithm of a number.

    Args:
    ----
        x: A number of type float. Must be positive.

    Returns:
    -------
        Natural logarithm of x (ln(x)) of type float.

    Raises:
    ------
        ValueError: If x is not positive.
    """
    if x <= 0:
        raise ValueError("Input must be positive for natural logarithm.")
    return math.log(x)


def mul(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        Product of a and b of type float.

    """
    return a * b


def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        Sum of a and b of type float.

    """
    return a + b


def id(a: float) -> float:
    """Returns an id unchanged.

    Args:
    ----
        a: A number of type float.

    Returns:
    -------
        A unchanged number of type float.

    """
    return a


def neg(a: float) -> float:
    """Negates a number.

    Args:
    ----
        a: A number of type float.

    Returns:
    -------
        A negated number of type float.

    """
    return float(-a)


def lt(a: float, b: float) -> bool:
    """Checks if one number is less than another.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        True if a < b, False otherwise.

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Checks if two numbers are equal.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        True if a == b, False otherwise.

    """
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        The maximum of a and b.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        True if |a - b| < 1e-2, False otherwise.

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        a: A number of type float.

    Returns:
    -------
        The sigmoid of a.

    """
    if a >= 0:
        z = math.exp(-a)
        return 1 / (1 + z)
    else:
        z = math.exp(a)
        return z / (1 + z)


def relu(a: float) -> float:
    """Applies the ReLU activation function.

    Args:
    ----
        a: A number of type float.

    Returns:
    -------
        The ReLU of a (max(0, a)).

    """
    return float(max(0.0, a))


def exp(a: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        a: A number of type float.

    Returns:
    -------
        e^a.

    """
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        a: A number of type float.

    Returns:
    -------
        1/a.

    """
    return 1 / a


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        b/a.

    """
    return b / a


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        -b/(a^2).

    """
    return -b / (a**2)


def relu_back(a: float, b: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        a: A number of type float.
        b: A number of type float.

    Returns:
    -------
        b if a > 0, else 0.

    """
    return b * (a > 0)


def map(fn: Callable[[float], float], lst: Iterator[float]) -> Iterator[float]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        fn: A function which takes a float parameter.
        lst: An iterator that yields float values.

    Returns:
    -------
        An iterator of transformed float values.

    """
    # results = []
    for x in lst:
        yield fn(x)

    # return results


def zipWith(
    fn: Callable[[float, float], float], lst1: Iterator[float], lst2: Iterator[float]
) -> Iterator[float]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        fn: A function that takes 2 float inputs.
        lst1: An iterator consisting of float values.
        lst2: An iterator consisting of float values.

    Returns:
    -------
        An iterator consisting of float values.

    """
    # results =[]
    for x, y in zip(lst1, lst2):
        yield fn(x, y)
    # /return results


def reduce(fn: Callable[[float, float], float], lst: Iterator[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        fn: A function that takes 2 float input values.
        lst: An iterator that yields float values.

    Returns:
    -------
        A single float value.

    """
    try:
        ans = next(lst)
    except StopIteration:
        return 0.0
    for x in lst:
        ans = fn(ans, x)

    return ans


def negList(lst: List[float]) -> List[float]:
    """Negate all elements in a list using map.

    Args:
    ----
        lst: List consists of float elements.

    Returns:
    -------
        A list of negated float elements.

    """
    new_iter = iter(lst)
    res_iter = map(neg, new_iter)
    res_list = list(res_iter)
    return res_list


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add corresponding elements from two lists using zipWith.

    Args:
    ----
        lst1: A list of float elements.
        lst2: A list of float elements.

    Returns:
    -------
        A list of float elements containing sums.

    """
    iter1 = iter(lst1)
    iter2 = iter(lst2)
    res_iter = zipWith(add, iter1, iter2)
    res_list = list(res_iter)
    return res_list


def prod(lst: List[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        lst: A list of float elements.

    Returns:
    -------
        A single float value representing the product.

    """
    new_iter = iter(lst)
    res = reduce(mul, new_iter)
    return res


def sum(lst: List[float]) -> float:
    """Calculate the sum of all elements in a list.

    Args:
    ----
        lst: A list of float elements.

    Returns:
    -------
        A single float value representing the sum.

    """
    new_iter = iter(lst)
    res = reduce(add, new_iter)
    return res


# ## Task 0.
# 3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
