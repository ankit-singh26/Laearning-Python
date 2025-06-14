import converter
import numpy as np

print(converter.km_to_miles(10))
print(converter.celsius_to_fahrenheit(25))

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

arr = np.array([5, 10, 15, 20, 25])
print("Average=", np.mean(arr))
print("Multiply=", arr * 2)

def get_matrix(n):
    print(f"Enter {n} (comma-separated rows):")
    rows = input().split(';')
    matrix = [list(map(int, row.split(','))) for row in rows]
    return np.array(matrix)

try:
    A = get_matrix("Matrix A")
    B = get_matrix("Matrix B")

    print("\nMatrix A:\n", A)
    print("Matrix B:\n", B)

    print("\nA + B =\n", A + B)
    print("\nA - B =\n", A - B)
    print("\nA * B (element-wise) =\n", A * B)
    print("\nA @ B (matrix multiplication) =\n", A @ B)

except ValueError as e:
    print("Invalid input:", e)
except Exception as e:
    print("Error:", e)



