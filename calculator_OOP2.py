# -*- coding: utf-8 -*-

import calculator_OOP as cal

class bad_calculator(cal.Calculator): #继承父类
    
    def add(self,x,y):
        print('sorry does not work') #新的类方法，替代了父类方法
        
        
        
bad_cal = bad_calculator(cal.Calculator.price,10,5) #调用父级的属性，注意在父类中默认的属性
print(bad_cal.price,bad_cal.name)

bad_cal.minus(20,7)
bad_cal.add(9,4)