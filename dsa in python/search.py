# from array import *

# def search(ar,target):
#     for i in range (len(ar)):
#         if ar[i]==target:
#             return i
#     return -1

# ar=array("i",[1,5,6,4,2,8])
# target=8
# if search(ar,target):
#     print("found at index",search(ar,target))
# else:
#     print("not found")
from array import *

def binarysearch(ar,n):
    start=0
    end=len(ar)-1
    while start <= end:
        
        mid=(start+end)//2
        if ar[mid]==n:
            print("found at index",mid)
            return mid
        elif ar[mid]<n:
            start=mid+1
        else:
            end=mid-1
    print("not found")
    return -1
ar=array("i",[1,2,3,4,5,6,7,8])
n=10
binarysearch(ar,n)
