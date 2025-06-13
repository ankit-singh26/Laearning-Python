nums = [5, 2, 9, 1, 7]
maximum = 0

for num in nums:
    if num > maximum:
        maximum = num

print(maximum)

countries = {
    "India": "Delhi",
    "Pakistan": "Lahore",
    "Afghanistan": "Kabul"
}

country = input("Enter the country: ")
capital = "Not found"

print(countries.get(country, "Not found"))

def reverse_list(lst):
    return lst[::-1]  # Simple slice-based reverse

print(reverse_list(nums))
