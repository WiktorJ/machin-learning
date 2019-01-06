import os

file1 = open('bonus_solution.txt', 'r')
file2 = open('bonus_solution2.txt', 'r')

line_n = os.stat("bonus_solution2.txt").st_size
print(line_n/2)

c = 0
for i in range(0, int(line_n/2)):
    if next(file1) == next(file2):
        c += 1

print(c/int(line_n/2))
