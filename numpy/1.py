import numpy as np

student= np.array([1,2,3,4,2,1,1,3,4,2,1,1,2,3,4,2,1,1])

total=0 
for i in student:
    if i==1:
        total+=1
print("Total number of students who scored 1:", total)