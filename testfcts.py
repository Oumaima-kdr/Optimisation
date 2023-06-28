# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:08:13 2021

@author: OumaimaKhadira
"""


import numpy as np
import math as ma
import matplotlib.pyplot as plt




#TEST FCT


def ackley(x,a=20,b=0.2,c=2*np.pi):
    x = np.array(x)
    d = len(x)
    
    sum1 = sum(x**2)
    sum2 = sum(np.cos(c*x))
    
    term1 = -a*np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)
    
    y = term1 + term2 + a +np.exp(1)
    
    return(y)



def eggholder(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    
    term1 = -(x2+47)*np.sin(np.sqrt(abs(x2+x1/2+47)))
    term2 = -x1*np.sin(np.sqrt(abs(x1-(x2+47))))
    
    y = term1+term2
    
    return(y)



def spheref(x):
    x = np.array(x)
    
    summ = sum(x**2)
    
    y = summ
    
    return(y)



def booth(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    
    term1 = (x1+2*x2-7)**2
    term2 = (2*x1+x2-5)**2
    
    y = term1+term2
    
    return(y)
        

def mccorm(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    
    term1 = np.sin(x1+x2)
    term2 = (x1-x2)**2
    term3 = -1.5*x1
    term4 = 2.5*x2
    
    y = term1+term2+term3+term4+1
    
    return(y)



def camel(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    
    term1 = 2*x1**2
    term2 = -1.05*x1**4
    term3 = x1**6/6
    term4 = x1*x2
    term5 = x2**2
    
    y = term1+term2+term3+term4+term5

    return(y)



def easom(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    
    fact1 = -np.cos(x1)*np.cos(x2)
    fact2 = np.exp(-(x1-np.pi)**2-(x2-np.pi)**2)
    
    y = fact1*fact2
    return(y)


def matyas(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    
    term1 = (x1**2 + x2**2)*0.26
    term2 = x1*x2*0.48
    
    y = term1-term2
    return(y)


def beal(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    
    term1 = (1.5-x1+x1*x2)**2
    term2 = (2.25-x1+x1*x2**2)**2
    term3 = (2.625-x1+x1*x2**3)**2
    
    y = term1+term2+term3
    return(y)

def himmelblau(x):
    x = np.array(x)
    
    x1 = x[0]
    x2 = x[1]
    term1=(x1**2+x2-11)**2
    term2=(x1+x2**2-7)**2
    y=term1 +term2
    
    return y
    


functions_names = ["ackley","eggholder","spheref","booth",
                   "mccorm","camel","easom","matyas","beal","himmelblau"]


testfunctions = {"ackley" : {"function": ackley, "domain": [[-32.768, 32.768]],
                             "global_min": 0, "at": 0,
                             "shape": "many local minima",
                             "property": "many local minima"},
                 
                 "eggholder" : {"function" : eggholder, "domain" : [[-512, 512]],
                                "global_min": -959.6407, "at": [512,404.2319],
                                "shape": "many local minima",
                                "property": "large number of local minima"},
                 "spheref" : {"function" : spheref, "domain" : [[-5.12, 5.12]],
                              "global_min": 0, "at": 0,
                              "shape": "bowl shaped",
                              "property": "convexe function"},
                 "booth" : {"function" : booth, "domain" : [[-10, 10]],
                            "global_min": 0, "at": [1,3],
                            "shape": "plate shaped",
                            "property": "RAS"},
                 "mccorm" : {"function" : mccorm, "domain" : [[-1.5, 4],[-3, 4]],
                             "global_min": -1.9133, "at": [-0.54719,-1.54719],
                             "shape": "plate shaped",
                             "property": "RAS"},
                 "camel" : {"function" : camel, "domain" : [[-5, 5]],
                            "global_min": 0, "at": 0,
                            "shape": "valley shaped",
                            "property": "3 local minima"},
                 "easom" : {"function" : easom, "domain" : [[-100, 100]],
                            "global_min": -1, "at": [np.pi,np.pi],
                            "shape": "steep ridges",
                            "property": "several local minima and the global min has a small area relative to the search space"},
                 "matyas" : {"function" : matyas, "domain" : [[-10, 10]],
                            "global_min": 0, "at": 0,
                            "shape": "plate shaped",
                            "property": "RAS"},
                 "beal" : {"function" : beal, "domain" : [[-4.5, 4.5]],
                            "global_min": 0, "at": [3,0.5],
                            "shape": "RAS",
                            "property": "sharp peaks at the corners of the domain"},
                 "himmelblau" : {"function" : himmelblau, "domain" :[[-5, 5]],
                                 "global_min": 0,  "at": [[3.0,2.0],
                                                          [-2.805118,3.131312],
                                                          [-3.779310,-3.283186],
                                                          [3.584428,-1.848126]],
                                 "shape": "many local minima",
                                  "property": "many local minima and 4 global minima"}
                 }



