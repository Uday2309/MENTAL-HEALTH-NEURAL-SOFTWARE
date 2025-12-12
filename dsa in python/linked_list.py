# class node:
#     def __init__(self, data):
#         self.data=data
#         self.next=None

# node1=node(10)
# node2=node(20)
# node3=node(30)
# node4=node(40)
# node5=node(50)

# node1.next=node2
# node2.next=node3
# node3.next=node4    
# node4.next=node5

#at begining 
# new_node=node(25)
# new_node.next=node1
# node1=new_node

#at mid
# new_node=node(35)
# new_node.next=node3.next
# node3.next=new_node

#at last
# new_tail=node(45)
# node5.next=new_tail


# node1 = node1
# while node1:
#     print(node1.data, end=" -> ")
#     node1 = node1.next
# print("None") 



#doubly linked list//////////


class dnode:
    def __init__(self,data):
        self.data=data
        self.next=None
        self.prev=None

node1=dnode(10)
node2=dnode(20) 
node3=dnode(30)
node4=dnode(40)
node5=dnode(50)


node1.next=node2
node2.prev=node1

node2.next=node3
node3.prev=node2

node3.next=node4
node4.prev=node3

node4.next=node5
node5.prev=node4



   
print("Traversal in forward direction:")
while node1:
    print(node1.data, end=" -> ")
    node1 = node1.next
print("None") 


print("\nTraversal in backward direction:")
while node5:
    print(node5.data, end=" -> ")
    node5 = node5.prev
print("None") 


#circular linked list//////////

