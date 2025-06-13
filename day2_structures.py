#Lists-Lists are ordered, mutable collections.
fruits = ["apple", "banana", "mango"]
print(fruits[0])
fruits.append("orange")
fruits.remove("banana")
print(len(fruits))

for fruit in fruits:
    print(fruit)

#Tuples-Tuples are like lists but immutable(cannot change). Why use tuples? Faster, safer for constant data (like coordinates).
point = (4, 5)
print(point[0])

x, y = point
print(x + y)

#Dictionaries-It stores key-value pairs.
student = {
    "name": "Ankit",
    "age": 21,
    "marks": [90, 85, 92]
}

print(student["name"])
student["college"] = "NIT Jamshedpur"

for key, value in student.items():
    print(key,"->", value)

#Functions-Reusable blocks of code.
def greet(name):
    return f"Hello, {name}!"

print(greet("Ankit"))

def power(x, y=2):
    return x ** y

print(power(3))
print(power(3,3))