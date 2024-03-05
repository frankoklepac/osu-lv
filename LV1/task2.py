class OutOfBoundsError(Exception):
    pass

try:
  grade=float(input("Enter your grade: "))
  if grade>1.0:
    raise OutOfBoundsError
  if grade>=0.9:
    print("A")
  elif grade>=0.8:
    print("B")
  elif grade>=0.7:
    print("C")
  elif grade>=0.6:
    print("D")
  else:
    print("F")

except OutOfBoundsError:
  print("Grade is out of bounds")

except ValueError:
  print("Grade is not a number")