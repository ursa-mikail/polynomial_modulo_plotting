"""
The issue may arises because the values selected for a, b, c, and p can be very large, causing precision errors and unexpected results in the calculations.
To avoid these issues, we need to carefully select values for a, b, c, and p such that they are within a manageable range. This will help ensure the computations remain accurate.
"""

import numpy as np
import sys
import random

def compute(poly, p, x):
    d = (int(poly[2]) + int(poly[1]) * x + int(poly[0]) * x * x) % p
    return int(d)

def coef(i, n):
    num = 1
    denom = 1
    for x in range(1, n + 1):
        if x != i:
            num *= x
            denom *= (i - x)
    return num / denom


import random

def random_int(bound_lower, bound_upper):
    """
    Returns a random integer between bound_lower and bound_upper (inclusive).

    :param bound_lower: The lower bound (inclusive)
    :param bound_upper: The upper bound (inclusive)
    :return: A random integer between bound_lower and bound_upper
    """
    return random.randint(bound_lower, bound_upper)


# List of possible primes
primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 2**7-1, 359, 659, 65537, 2**13-1, 2**17-1, 2**19-1, 2**31-1, 2**61-1, 2**89-1, 2**107-1, 2**127-1, 2**255-19]

# List of polynomial equations
polynomials = [
    (7, 19, 21),
    (21, 19, 21),
    (500, 499, 78),
    (9999, 7654, 501)
]

# Randomly select a prime and polynomial coefficients
p = random.choice(primes)
a, b, c = random.choice(polynomials)

lower_bound = 0
upper_bound = 2**128

#a, b, c = (random_int(lower_bound, upper_bound), random_int(lower_bound, upper_bound), random_int(lower_bound, upper_bound))

while (p <= a):
  p = random.choice(primes)

A = [c, b, a]

f_1 = compute(A, p, 1)
f_2 = compute(A, p, 2)
f_3 = compute(A, p, 3)

print(f"Selected Polynomial: f(x) = {a} + {b}x + {c}x^2 (mod {p})\n")

print(f"Polynomial Coefficients:\n{np.poly1d(A)}")
print(f"Prime: {p}\n")

f_0 = int(((f_1) * coef(1, 3) + (f_2) * coef(2, 3) + (f_3) * coef(3, 3)) % p)

print(f"Shares: f_1={f_1}, f_2={f_2}, f_3={f_3}")
print(f"Lagrange Interpolation constants:\nc_1={coef(1, 3)}, c_2={coef(2, 3)}, c_3={coef(3, 3)}\n")
print(f"Recovered secret: f_0 (secret)={f_0}")
if f_0 == A[2]:
    print("Successful recovery")
else:
    print(f"Failed recovery: f_0: {f_0} and a: {A[2]}")


"""
Selected Polynomial: f(x) = 21 + 19x + 21x^2 (mod 65537)

Polynomial Coefficients:
    2
21 x + 19 x + 21
Prime: 65537

Shares: f_1=61, f_2=143, f_3=267
Lagrange Interpolation constants:
c_1=3.0, c_2=-3.0, c_3=1.0

Recovered secret: f_0 (secret)=21
Successful recovery
"""

# Example with an explicitly larger constant term
a = 2**270  # Larger coefficient
b = 2**265  # Larger coefficient
c = 2**260  # Larger coefficient

# Construct polynomial
A = [c, b, a]

# Compute shares for x = 1, 2, 3 (again using the chosen prime modulus)
f_1 = compute(A, p, 1)
f_2 = compute(A, p, 2)
f_3 = compute(A, p, 3)

print(f"Selected Polynomial: f(x) = {a} + {b}x + {c}x^2 (mod {p})\n")
print(f"Polynomial Coefficients:\n{np.poly1d(A)}")
print(f"Prime: {p}\n")

# Use Lagrange interpolation to recover the secret
f_0 = int(((f_1) * coef(1, 3) + (f_2) * coef(2, 3) + (f_3) * coef(3, 3)) % p)

print(f"Shares: f_1={f_1}, f_2={f_2}, f_3={f_3}")
print(f"Lagrange Interpolation constants:\nc_1={coef(1, 3)}, c_2={coef(2, 3)}, c_3={coef(3, 3)}\n")
print(f"Recovered secret: f_0 (secret)={f_0}")
if f_0 >= 2**256:
    print("Successful recovery: f_0 is greater than or equal to 2^256")
else:
    print(f"Failed recovery: f_0 is {f_0}, less than 2^256")

"""
Selected Polynomial: f(x) = 1897137590064188545819787018382342682267975428761855001222473056385648716020711424 + 59285549689505892056868344324448208820874232148807968788202283012051522375647232x + 1852673427797059126777135760139006525652319754650249024631321344126610074238976x^2 (mod 65537)

Polynomial Coefficients:
           2
1.853e+78 x + 5.929e+79 x + 1.897e+81
Prime: 65537

Shares: f_1=16912, f_2=17472, f_3=18064
Lagrange Interpolation constants:
c_1=3.0, c_2=-3.0, c_3=1.0

Recovered secret: f_0 (secret)=16384
Failed recovery: f_0 is 16384, less than 2^256
"""