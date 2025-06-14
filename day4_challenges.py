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


