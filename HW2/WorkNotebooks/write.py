#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Write to the output File
def writeOutput(result, fileName="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])
    with open(fileName, 'w') as f:
        f.write(res)
        


# In[4]:


# Write the "PASS" action to output File
def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")
    


# In[5]:


# Write the input action to the output File to feed to host.py
def writeNextInput(piece_type, previous_board, board, fileName="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    
    # get previous board config
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"
        
    # add it to the current board config    
    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"
    
    # write it all to the file, starting from the rear of res
    with open(fileName, 'w') as f:
        f.write(res[:-1]);
        
    

