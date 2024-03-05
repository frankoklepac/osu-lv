def total_euro(hours, pay):
    return hours * pay

hours = int(input("Input working hours:"))
pay = float(input("Input pay per hour:"))

print("Your salary is:", total_euro(hours, pay), "euros.")
