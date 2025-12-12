class stack:

    def __init__(self):
        self.stack=[]

        def push(self,data):
            self.stack.append(data)

        def pop(self):
            if sel.is_empty():
                return "stack is empty"
            return self.stack.pop()
        
        def peek(self):
            if self.is_empty():
                 return ("stack is empty") 
        return self.stack[-1]
    

        def is_empty(self):
            return self.top is None
        
        def display(self):
            if self.is_empty():
                return "stack is empty"
            for i in range(len(self.stack)-1,-1,-1):
                print(self.stack[i],end=" ")
            print()