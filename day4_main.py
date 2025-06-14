import mymath

print(mymath.square(4))
print(mymath.cube(3))

#Exception Handling
try:
    a = int(input("Enter a number: "))
    b = int(input("Enter another number: "))
    print("Result =", a/b)
except ZeroDivisionError:
    print("You can't divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Done")

#NumPy = Numerical Python – foundation of ML data handling.
import numpy as np

arr = np.array([1, 2, 3, 4])
print("Array:", arr)
print("Mean:", np.mean(arr))
print("Sum:", np.sum(arr))
print("Squared:", arr**2)

a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

print("1D Array:", a)
print("2D Array:\n", b)

print(a[0])      
print(b[1][1]) 

print(a[1:])    # [2 3]
print(b[:, 1])  # second column => [2 4]

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

# Element-wise
print(x + y)
print(x * y)

# Matrix multiplication
print(x @ y)        # or np.dot(x, y)

z = np.array([1, 2])
print(x + z)
# Output: [[2 4], [4 6]]

print("Sum =", np.sum(x))
print("Mean =", np.mean(x))
print("Standard Deviation =", np.std(x))