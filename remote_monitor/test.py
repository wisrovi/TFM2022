
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
c = a*b
d = np.dot(a,b)

print(c)
print(d)
print(b[0])

import sys
sys.exit()

class Units_vel:
    kph = "km/h"
    mph = "mph"


class Car:
    def __init__(self, max_vel:int, units:Units_vel=Units_vel.kph):
        self.max_vel = max_vel
        self.units = units

    def __str__(self):
        return f"Car with the maximum speed of {self.max_vel} {self.units}"


class Boat:
    def __init__(self, knots:int):
        self.knots = knots

    def __str__(self):
        return f"Boat with the maximum speed of {self.knots} knots"


print(Car(120, Units_vel.kph))
print(Boat(82))


DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def missingCharacters(s:str):
    # Write your code here
    rta = ""
    to_array = [char for char in s]

    group_letter = list()
    group_number = list()
    for char in to_array:
        if char in LETTERS:
            group_letter.append(char)
        else:
            group_number.append(char)

    for number in [num for num in DIGITS]:
        if number not in group_number:
            rta += number

    for letter in [let for let in LETTERS]:
        if letter not in group_letter:
            rta += letter

    return rta


s = "7985interdisciplinario12"
rta = missingCharacters(s)
print(rta)
print("0346bfghjkmoquvwxz")









def findSum(numbers, queries):
    acum_numbers = [0]
    acum_zeros = [0]
    for number in numbers:
        acum_numbers.append(acum_numbers[-1] + number)
        acum_zeros.append(acum_zeros[-1] + (number == 0))
    #print(acum_numbers, acum_zeros)
    return [
        acum_numbers[end] - acum_numbers[init - 1] + x * (
                acum_zeros[end] - acum_zeros[init - 1]
        )
        for init, end, x in queries]





numbers = [5, 10, 10]
queries = [[1, 2, 5]]
print(findSum(numbers, queries), 15)


def mostBalancedPartition(parent, files_size):
    n = len(parent)
    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    size_sums = [None for _ in range(n)]

    def size_sums_rec(i):
        size_sums[i] = files_size[i] + sum(size_sums_rec(c) for c in children[i])
        return size_sums[i]

    size_sums_rec(0)
    return min(abs(size_sums[0] - 2 * ss) for ss in size_sums[1:])



























def renameFile(newName, oldName):
    size_new_name = len(newName)
    size_old_name = len(oldName)
    dp = [1 for j in range(size_old_name + 1)]
    for i in range(1, size_new_name + 1):
        dpp = [0 for _ in range(size_old_name + 1)]
        for j in range(i, size_old_name + 1):
            dpp[j] = dpp[j - 1]
            if newName[i - 1] == oldName[j - 1]:
                dpp[j] += dp[j - 1]
        dp = dpp
    return dp[-1] % (10**9 + 7)


class Node:
    def __init__(self, parent, l, r, op=max):
        self.parent = parent
        self.l = l
        self.r = r
        self.lc = None
        self.rc = None
        self.val = r - l
        self.op = op

    def split(self, x):
        # No balancing, but doesn't seem to give timeouts.
        assert self.l <= x <= self.r
        if x == self.l or x == self.r:
            # Split lies on borders.
            return
        if self.lc:
            if x == self.lc.r:
                # Split lies on mid split.
                return
            if x < self.lc.r:
                self.lc.split(x)
            else:
                self.rc.split(x)
            self.val = self.op(self.lc.val, self.rc.val)
        else:
            self.lc = Node(parent=self, l=self.l, r=x)
            self.rc = Node(parent=self, l=x, r=self.r)
            self.val = self.op(x - self.l, self.r - x)


def getMaxArea(w, h, isVertical, distance):
    w_root = Node(parent=None, l=0, r=w)
    h_root = Node(parent=None, l=0, r=h)
    ans = []
    for iv, d in zip(isVertical, distance):
        if iv:
            w_root.split(d)
        else:
            h_root.split(d)
        ans.append(w_root.val * h_root.val)
    return ans



















from calendar import *
year = 2023
print(calendar(year, 2,1,8,4))


from langdetect import detect
text = "hola mundo, como estas"
print(detect(text))