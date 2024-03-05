num_list = []

while(True):
    number = input("Input a number: ")
    try:
      if number == "Done":
          break
      num_list.append(float(number))

    except ValueError:
        print("Input is not a number")
        continue
    
print("Maximum number is: ", max(num_list))
print("Minimum number is: ", min(num_list))
print("Average number is: ", sum(num_list)/len(num_list))
print("Number of numbers: ", len(num_list))