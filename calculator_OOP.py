class Calculator:
   #设置类属性: 所有属于Calculator类的对象object自带的默认属性
    price = 20
    
####################类方法###################################
    
    def __init__(self, price, height, width, name='simple calculator'):
    #初始一个object,所有的对象的self都带有上述指定属性，注意是否默认
        self.name = name
        self.price = price
        self.hi = height
        self.wi = width
        
    def add(self,x,y):
        print(x+y)
        
    def minus(self,x,y):
        print(x-y)
        
    def multiply(self,x,y):
        print(x*y)
        
    def divide(self,x,y):
        print(x/y)

######################建立对象####################################      
c = Calculator(Calculator.price,12,14) 
#创建对象c时调用了类属性,要为非默认属性的赋值
d = Calculator(2,9,9,'working calcualtor')
#创建对象d时 设立了新的属性
print(c.name,c.price)
print(d.name,d.price)

c.minus(10,5)
c.add(9,5)





