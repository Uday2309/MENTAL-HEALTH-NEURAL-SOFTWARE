# # n=5
# # for i in range (n):
# #      for j in range (n):
# #         print("*",end=" ")
# #      print()

# n=5 
# for i in range (n):
#     for j in range (i+1):
#         print("*",end=" ")
#     print()

n=5 
for i in range(n):
    for j in range(i,n):
        print(" ",end=" ")
    for j in range(i+1):
        print("*",end=" ")
    for j in range(i+0):
        print("*",end=" ")
    for j in range(i,n):
        print(" ",end=" ")
    for j in range(i,n):
        print(" ",end=" ")
    for j in range(i+1):
        print("*",end=" ")
    for j in range(i+0):
        print("*",end=" ")
    print()
