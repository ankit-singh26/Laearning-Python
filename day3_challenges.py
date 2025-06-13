str = input("Enter the string: ")
with open("log.txt", "w") as f:
    f.write(str)

people = [("Ankit", 21), ("Ravi", 19), ("Simran", 22)]
people.sort(key=lambda age: age[1])
print(people)

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return self.length * self.width
    
    def perimeter(self):
        return 2 * (self.length + self.width)
    
rec = Rectangle(2, 3)
print(rec.area())
print(rec.perimeter())

