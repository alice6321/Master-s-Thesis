# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:29 2020

@author: Administrator
"""

#均勻分布
import numpy as np

 

def Dataset1(t,x,y):
    #np.random.seed(None)#取消固定隨機數種子
    np.random.seed(t)
    a = np.random.rand(x, y) * 100
    #x = round(random.uniform(0, 100)) # 產生介於 2.5 到 10 之間的隨機浮點數（2.5 <= x < 10.0）
    return a
    

    

