n = int(input("Enter a number: "))

if n % 2 == 1:
    print("The number is Odd.")
else:
    print("The number is Even.")

sum = 0

for i in range(1, n + 1):
    sum += i

print("Sum =", sum)

for i in range(1, 11):
    print(n,"x",i,"=",n*i)

fact = 1
for i in range(1, n + 1):
    fact *= i
print("Factorial of", n, "is", fact)