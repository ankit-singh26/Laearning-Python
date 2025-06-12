#variables
name = "Ankit"
age = 21
height = 5.10
is_student = True

print(name, age, height, is_student)
print(type(name), type(age), type(height), type(is_student))

#input
user_name = input("Enter your name: ")
print("Hello", user_name)

#Type-conversion
num = int(input("Enter a number: "))
print("square is:", num*num)

#conditional statements 
Age = int(input("Enter your age: "))

if Age < 18:
    print("You are a minor.")
elif Age < 60:
    print("You are an adult.")
else:
    print("You are a senior citizen.")

#Loops
for i in range(1, 6):
    print("Count:", i)

count = 1
while count <= 5:
    print("While Count:", count)
    count += 1