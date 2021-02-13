#!/usr/bin/env python
# coding: utf-8

# In[10]:


# read the input from the other player - also a way of seeing everything on the board
def readInput(n, fileName="input.txt"):

    with open(fileName, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board


# In[11]:


def readOutput(fileName="output.txt"):
    with open(fileName, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y


# In[ ]:




