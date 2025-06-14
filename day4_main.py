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