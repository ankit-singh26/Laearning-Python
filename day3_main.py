# File Handling in Python
with open("sample.txt", "w") as f:
    f.write("Hello, this is Day 3!\n")
    f.write("Python is awesome.")

with open("sample.txt", "r") as f:
    content = f.read()
    print(content)

with open("sample.txt", "a") as f:
    f.write("\nThis is a new line.")

# Lambda Functions
add = lambda x, y: x + y
print(add(3, 4))

squares = list(map(lambda x: x * x, [1, 2, 3, 4]))
print(squares)

evens = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5]))
print(evens)

# OOP - Classes and Objects
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name}."

p1 = Person("Ankit", 22)
print(p1.greet())