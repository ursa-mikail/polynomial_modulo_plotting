import numpy as np
import sys
import random

def compute(poly, p, x):
    x = np.asarray(x)  # Ensure x is a numpy array for vectorized operations
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


a, b, c = (random.choice(primes), random.choice(primes), random.choice(primes))

while (p <= a):
  p = random.choice(primes)

A = [c, b, a]

f_1 = compute(A, p, 1)
f_2 = compute(A, p, 2)
f_3 = compute(A, p, 3)

max_y = max([f_1, f_2, f_3])

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

#
import numpy as np
import matplotlib.pyplot as plt
# Plotting the polynomial and points
x = np.linspace(-1, 5, 400)
y = (c * x**2 + b * x + a) % p

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'Polynomial f(x) = {a} + {b}x + {c}x^2 (mod {p})')
plt.scatter([0, 1, 2, 3], [f_1, f_2, f_3, f_0], color='red', zorder=5)
plt.text(0, f_0, f'({0}, {f_0})', fontsize=15, ha='right')
plt.text(1, f_1, f'({1}, {f_1})', fontsize=12, ha='right')
plt.text(2, f_2, f'({2}, {f_2})', fontsize=12, ha='right')
plt.text(3, f_3, f'({3}, {f_3})', fontsize=12, ha='right')

plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title('Polynomial and Chosen Points')
plt.xlabel('x')
plt.ylabel('f(x) % p')
plt.legend()
plt.show()


# Given points
x_points = [0, 1, 2, 3]
y_points = [f_0, f_1, f_2, f_3]

# Find polynomial coefficients that exactly match points
coeffs = np.polyfit(x_points, y_points, 2)  # 2 for quadratic

# Calculate polynomial values
y = np.polyval(coeffs, x) % p

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'Interpolating Polynomial')
plt.scatter(x_points, y_points, color='red', zorder=5)

"""
Selected Polynomial: f(x) = 29 + 2305843009213693951x + 659x^2 (mod 71)

Polynomial Coefficients:
     2
659 x + 2.306e+18 x + 29
Prime: 71

Shares: f_1=67, f_2=3, f_3=50
Lagrange Interpolation constants:
c_1=3.0, c_2=-3.0, c_3=1.0

Recovered secret: f_0 (secret)=29
Successful recovery
"""
