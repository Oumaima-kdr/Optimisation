# coding utf:8
from logging import error
import numpy as np
import math as ma
import matplotlib.pyplot as plt 
import numpy.random as rn
import numpy as np
import time
import math

#Library python ABC 
#https://pypi.org/project/beecolpy/
#Step-by-step:
from beecolpy import abc

#Vectorization python des fonction

class testFunctionOptimization:
    
    def __init__(self):
        pass
 
    #methode ackley function
    def ackleyFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
    
        return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20
    

    #methode sphere function
    def sphereFunction(self,X):
        
        return np.sum(X)
    
    #methode Rosenbrock function
    def Rosenbrock(self,X):
        array=np.asarray(X)
        n=array.shape[0]
        f=0
        for i in np.arange(1,n):
            f=f+((100*((array[i]-array[i-1])**2))+(1-array[i-1])**2)
            
        return f
    
    #methode Beal Function
    
    def bealFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return (1.5-x+x*y)**2+(2.25-x+x*y**2)**2+(2.625-x+x*y**3)**2
    
    #methode goldstein Price Function
    
    def goldsteinPriceFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    
    #methode booth Function
    
    def boothFunction(self,X):
        
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return (x+2*y-7)**2+(2*x+y-5)**2
    
    
    #methode bukin Function N6
    
    
    def bukinFunctionN6(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return 100*np.sqrt(np.fabs(y-0.01*x**2))+0.01*np.fabs(x+10)
    
    #methode matyas Function
    
    def matyasFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return 0.26*(x**2+y**2)-0.48*x*y
    
    #methode himmelblau Function
    
    def himmelblauFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return (x**2+y-11)**2+(x+y**2-7)**2
    
    #methode ThreeHump Camel Function
    
    def ThreeHumpCamelFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return 2*x**2-1.05*x**4+x**6/6+x*y+y**2
    
    #methode Easom Function
    
    def EasomFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2+(y-np.pi)**2))
    
    #methode CrosI in Tray Function
    
    def CrosIinTrayFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return -0.0001*(np.fabs(np.sin(x)*np.cos(y)*np.exp(np.fabs(100-(np.sqrt(x**2+y**2)/np.pi))))+1)**0.1
    
    #methode  Eggholder Function
    def HolderTableFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return -np.fabs(np.sin(x)*np.cos(y)*np.exp(np.fabs(1-(np.sqrt(x**2+y**2)/np.pi))))

                        
    #methode Levi13 Function
    def Levi13Function(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
                        
        return np.sin(3*np.pi*x)**2+((x-1)**2)*(1+np.sin(3*np.pi*y)**2)+((y-1)**2)*(1+np.sin(2*np.pi*y)**2)
 

   
    #method Rastrigin Function
    def rastrigin(self,X):
        A = 10
        return A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])
    
    #methode  Eggholder Function   

    def EggholderFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        
        return (-(y + 47) * np.sin(np.sqrt(abs(x/2 + (y + 47)))) -x * np.sin(np.sqrt(abs(x - (y + 47)))))
    
        
        
        
    #method Styblinski–Tang Function
    def Styblinski_Tang_Function(self,X):
        return sum([(x**4-16*x**2+5*x)/2 for x in X])
    
    
    #method McCormick Function
    def McCormickFunction(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        return np.sin(x+y)+(x-y)**2-1.5*x+2.5*y+1 
    

    
    
   #method Schaffer2 Function
    def Schaffer2Function(self,X):
        x=np.asarray(X[0])
        y=np.asarray(X[1])
        return 0.5+(np.sin(x**2-y**2)**2-0.5)/(1+0.001*(x**2+y**2))**2
    
    

    
#     #method Schaffer2 Function
#     def Schaffer2Function(X):
#         x=np.asarray(X[0])
#         y=np.asarray(X[1])
#         return 0.5+(np.cos**2*(np.sin**2*(np.fabs(x**2-y**2)))-0.5)/(1+0.001*(x**2+y**2))**2
    
    
    
        

    
x = np.linspace(-6,6,30 )
y = np.linspace(-6,6,30)

#Bibliotheque of function

f=testFunctionOptimization()

#Meshgrid

X, Y = np.meshgrid(x, y)
Z=f.himmelblauFunction([X,Y])

functionName='himmelblau'
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 75, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('3D Representation of {}'.format(functionName))
ax.set_zlabel('z');



#
#uniform law with a b 
def random_start(interval):
    a, b = interval
    return a + (b - a) * rn.random_sample()

##q of the probabilities from interests

def q(fonction_s):
    if fonction_s >=0:
        return 1/(1+fonction_s)
    else :
        return 1+np.abs(fonction_s)

class ABC():
    
    '''Artificial bee colony'''
    
    def __init__(self,nbr_iteration,Nonlooker,food_Sources,fonction_cost,search_space,D=0.5,debug=True,overwrite=True):
        
        
        self.nbr_iteration=nbr_iteration
        self.Nonlooker=Nonlooker
        self.food_Sources=food_Sources
        self.fonction_cost=fonction_cost
        self.search_space=search_space
        self.D=D
        self.debug=True
        self.overwrite=True
        self.s=[]
        self.s_prime=[]
        self.sxi=np.zeros(Nonlooker)
        self.sxi_prime=np.zeros(Nonlooker)
        self.sn=[]
        self.sn_prime=[]
        self.e=[]
        self.compt=0
        self.fonction_cost_values=[]
        self.prob=np.zeros(food_Sources)
        self.w=np.zeros(Nonlooker)
        self.w_prime=np.zeros(Nonlooker)
        self.s_stars=[]
        self.s_stars_prime=[]
        self.costs=[]
        self.Truecost=0
        self.MSEs=[]
        self.period=0
        self.cost=0
        
        
    # food_Sources are the zones in the search space(S)
    #mathematical S is the zones of foods sources 
        
    def init_food_source(self):
        for i in range(self.food_Sources):#for each zone do 
            s1=random_start(self.search_space)
            s2=random_start(self.search_space)
            self.s.append(s1)
            self.s_prime.append(s2)#for have different random values 
            self.fonction_cost_values.append(self.fonction_cost([s1,s2]))
            self.e.append(0)

        index_max_fonction_cost_values=self.fonction_cost_values.index(max(self.fonction_cost_values))
        self.s_star=self.s[index_max_fonction_cost_values]
        self.s_star_prime=self.s_prime[index_max_fonction_cost_values]
        
            
        
    #Employed Bees Leave the Hive to Exploit Food Sources
    #Computation of new food sources
        
    def Bees_employed(self) :       
        v=self.s
              
        for i in range(self.food_Sources):
            sn1=[random_start((0,max(1,(self.food_Sources)))) for x in range(self.food_Sources) if random_start((0,max(1,(self.food_Sources))))!=self.s[i]]
            sn2=[random_start((0,max(1,(self.food_Sources)))) for x in range(self.food_Sources) if random_start((0,max(1,(self.food_Sources))))!=self.s[i]]
            
            k=int(random_start((1,self.D)))#choose the modified coordinate
            v_prime=self.s_prime #for define the y value like our function is two dimenssion function
            #Mutate the solution
            v[k]=self.s[k]+random_start((-1,1))*(self.s[k]-sn1[k])
            v_prime[k]=self.s_prime[k]+random_start((-1,1))*(self.s_prime[k]-sn2[k])
            
        for i in range(self.food_Sources): # The new solution is more interesting 
            if self.fonction_cost([v[i],v_prime[i]])<self.fonction_cost([self.s[i],self.s_prime[i]]):
                self.s[i]=v[i]
                self.s_prime[i]=v_prime[i]
                self.e[i]=0
                if self.fonction_cost([v[i],v_prime[i]])<self.fonction_cost([self.s_star,self.s_star_prime]):# The best solution is improved 
                    self.s_star=max(self.search_space[0],min(self.search_space[1],v[i]))
                    self.s_star_prime=max(self.search_space[0],min(self.search_space[1],v_prime[i]))
            else :
                self.e[i]=self.e[i]+1 # The new solution is worse
        
        self.MSEcost=self.fonction_cost([self.s_star,self.s_star_prime])
        #MSE 
        self.MSE=(abs(self.MSEcost-self.Truecost))**2
        self.MSEs.append(self.MSE)
        self.period=self.period+1

    def probability_of_interest(self):#Compute the probabilities from interests
        somme=0
        for i in  range(self.food_Sources):
            somme =somme +q(self.fonction_cost([self.s[i],self.s_prime[i]]))
            
        
        for i in range(self.food_Sources):
            self.prob[i]=(q(self.fonction_cost([self.s[i],self.s_prime[i]]))/somme)


    def onlooker(self):
        #Onlookers exploit the food sources 
        
        for i in range(self.Nonlooker):
            x1=np.random.choice(range(0,self.food_Sources),p=self.prob)
            x2=np.random.choice(range(0,self.food_Sources),p=self.prob)
            
            sn1=[random_start((0,max(1,(self.food_Sources)))) for x in range(self.food_Sources) if random_start((0,max(1,(self.food_Sources))))!=self.s[x1]]
            sn2=[random_start((0,max(1,(self.food_Sources)))) for x in range(self.food_Sources) if random_start((0,max(1,(self.food_Sources))))!=self.s[x2]]
            
            
            k=int(random_start((1,self.D)))#choose the modified coordinate
            
            #for define the y value like our function is two dimenssion function
            self.w[i]=self.s[x1-1]
            self.w_prime[i]=self.s_prime[x2-1] 

            
            self.sxi[i]=(self.s[x1-1])

            self.sxi_prime[i]=(self.s[x2-1])
        
            #Mutate the solution
            self.w[k]=self.s[k]+random_start((-1,1))*(self.s[k]-sn1[k])
            self.w_prime[k]=self.s_prime[k]+random_start((-1,1))*(self.s_prime[k]-sn2[k])

                    #Update the food sources and their counters */
        for i in range(self.Nonlooker):
            exi=np.zeros(self.Nonlooker)
            if self.fonction_cost([self.w[i],self.w_prime[i]])<self.fonction_cost([self.sxi[i],self.sxi_prime[i]]):
                exi[i]=0
                self.sxi[i]=self.w[i]
                self.sxi_prime[i]=self.w_prime[i]
                
                
                if self.fonction_cost([self.w[i],self.w_prime[i]])<self.fonction_cost([self.s_star,self.s_star_prime]):# The best solution is improved 
                    self.s_star=max(self.search_space[0],min(self.search_space[1],self.w[i]))
                    self.s_star_prime=max(self.search_space[0],min(self.search_space[1],self.w_prime[i]))
                    
            else :
                exi[i]=exi[i]+1 # The new solution is worse 

        #second period 
        MSE=(abs(self.MSEcost-self.Truecost))**2
        self.MSEs.append(MSE)
        period=self.period+1
        
    def scoot_bees(self):
        # The number of scout bees transformed into employed bees
        n=0
        c=[i for i in range(self.food_Sources) if self.e[i]>=max(self.e)]
        while n<self.Nonlooker and len(c)!=0:
            ej=[self.e[i] for i in c]
            i=ej.index(max(ej))
            s1=random_start(self.search_space)
            s2=random_start(self.search_space)
            if self.fonction_cost([s1,s2])<self.fonction_cost([self.s_star,self.s_star_prime]):# The best solution is improved 
                self.s_star=max(self.search_space[0],min(self.search_space[1],s1))
                self.s_star_prime=max(self.search_space[0],min(self.search_space[1],s2))
            c=[i for i in range(self.food_Sources) if self.e[i]>=max(self.e)]
            n=n+1
        self.cost=self.fonction_cost([self.s_star,self.s_star_prime])
        self.costs.append( self.cost)
        self.s_stars.append(self.s_star)
        self.s_stars_prime.append(self.s_star_prime)
        MSE=(abs(self.MSEcost-self.Truecost))**2
        self.MSEs.append(MSE)
        self.period=self.period+1
            




#
number_1=1
number_2=2

print(f"Readme\n The {number_1}st function is my own function abc \n the second one  function bellow is the {number_2} nd part function for testing abc algorithm ")

fonction=testFunctionOptimization()
method_function=fonction.EggholderFunction
abc_=ABC(200,100,100,method_function,(-512,512),D=0.5,debug=True,overwrite=True) 
abc_.init_food_source() 
nb_eval=0

while nb_eval!=abc_.nbr_iteration:  
    abc_.Bees_employed()
    abc_.probability_of_interest()
    abc_.onlooker()
    abc_.scoot_bees()
    if abc_.overwrite:
            
        for _ in range(20):
            print('\r {}'.format(_),end='    ')
            time.sleep(0.01)
            print('\r    ',end='')

    real_cost=abc_.cost
        
    if abc_.debug:print(" \n Step #{}/{} , x_state = {}, y_state = {} ,cost={}...".format(nb_eval, abc_.nbr_iteration, abc_.s_star, abc_.s_star_prime,abc_.cost))
    real_cost=abc_.cost

    if (abc_.s_star==min(abc_.search_space[0],abc_.search_space[1]) or abc_.s_star==max(abc_.search_space[0],abc_.search_space[1])) or ( (abc_.s_stars_prime==min(abc_.search_space[0],abc_.search_space[1]) or abc_.s_stars_prime==max(abc_.search_space[0],abc_.search_space[1]))):
        nb_eval=abc_.nbr_iteration
    else:
        nb_eval=nb_eval+1
 

    
            
        
    
 

#methode ackley function
fonction=testFunctionOptimization()
method_function=fonction.EggholderFunction
abc_obj = abc(method_function,boundaries=[(-512,512), (-512,512)],colony_size=100,scouts=0.5,iterations=200,min_max='min',nan_protection=True,log_agents=True)

#Execute algorithm: 
abc_obj.fit()

#Get solution obtained after fit() execution:

solution = abc_obj.get_solution()
fonction_=testFunctionOptimization()
method_function_=fonction_.EggholderFunction(solution)
print('solutions outcomes after using  library python algorithm abc ')
print(f"solution ={solution} cost = {method_function_}")


#Error clerk for our ower algorithme
cost_value_for_min=-959.6407
error__=abs(cost_value_for_min-real_cost)

print("Error clerk for our own algorithm : {:.3f}".format(error__))

#Error clerk for algorithme abc for library python
error_=abs(cost_value_for_min-method_function_)

print("Error clerk for  library python algorithm abc :{:.3f}".format(error_))