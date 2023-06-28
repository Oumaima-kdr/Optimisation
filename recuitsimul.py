# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:45:00 2021

@author: OumaimaKhadira
"""

import testfcts
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
    
    
    
# objective function
def objective(function_name,x):
	return testfcts.testfunctions[function_name]["function"](x)
 
#cette fct objective n'est pas utilisée dans la suite





#graphe fonction test

def graph(function_name):
    "plot de la fonction test sur le domaine de recherche associé"
    if function_name not in testfcts.functions_names:
      raise TypeError("Expected str in the list:"+str(testfcts.functions_names))

    domain = testfcts.testfunctions[function_name]["domain"]
    if len(domain) == 2:
        x = domain[0]
        y = domain[1]

        xaxis = np.arange(x[0], x[1], 0.1)
        yaxis = np.arange(y[0], y[1], 0.1)

        x, y = np.meshgrid(xaxis, yaxis)
    else:
        r_min, r_max = domain[0][0], domain[0][1]
        xaxis = np.arange(r_min, r_max, 0.1)
        yaxis = np.arange(r_min, r_max, 0.1)
        
        x, y = np.meshgrid(xaxis, yaxis)

    results = testfcts.testfunctions[function_name]["function"]([x, y])   #pas oublier les [] !!
        
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    plt.title("Graphe de la fct "+str(function_name))

    plt.show()


graph("himmelblau")
graph("matyas")
graph("eggholder")
graph("ackley")

graph("beal")
graph("camel")
graph("mccorm")

graph("ackley")
graph("easom")






#recuit simulé

def simulated_annealing(function_name, bounds, max_iterations, step_size, t0, tolerance):
    temp = [t0]
    nb_it = 0
    points = []
    values = []
    objective = testfcts.testfunctions[function_name]["function"]
    true_min = testfcts.testfunctions[function_name]["global_min"]
    if len(bounds) == 1:
        bounds = np.array([bounds[0],bounds[0]])
	# point initial
    best = bounds[:, 0] + rn.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluation de la fonction à en ce point
    best_eval = objective(best)
	
    curr, curr_eval = best, best_eval
	
    for i in range(max_iterations):
        points.append(best)
        values.append(best_eval)
		# voisin
        candidate = curr + rn.randn(len(bounds)) * step_size
		# evaluation du voisin
        candidate_eval = objective(candidate)
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
		# variation
        diff = candidate_eval - curr_eval
		# temperature
        #t = t0/(np.log(1+i+1))   #décroissance logarithmique
        t = t0*np.exp(-np.sqrt(i+1))  #décroissance exp
        temp.append(t)
        #t = t0*0.85**(i+1)   #décroissance geometrique
		# critere de Metropolis
        metropolis = np.exp(-diff / t)
        if diff < 0 or rn.rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval
        if abs(best_eval-true_min) > tolerance:
            nb_it = nb_it + 1
            
    return({"best": best, "best_eval": best_eval,
            "points": points, "values": values,
            "nb_iterations": nb_it, "temperature": temp})






fonction="ackley"
bounds = testfcts.testfunctions[fonction]["domain"]

test = simulated_annealing("ackley", bounds, 1000, 0.6, 100, 1e-1)



x=np.arange(0,505,5)
y1=10/(np.log(1+x+1))
y2=10*np.exp(-np.sqrt(x+1))
y3=10*0.85**(x+1)
plt.plot(x,y1, label="logarithmique")
plt.plot(x,y2, label="exponentiel")
plt.plot(x,y3, label="géométrique")
plt.xlabel("Itérations")
plt.ylim(0,5)
plt.legend()
plt.show()



#taux de succès en fonction d'une tolerance

def rate_sa(n,function_name,max_iterations,step_size,t0,tolerance):
    res = np.array([])
    nb_it = np.array([])
    true_min = testfcts.testfunctions[function_name]["global_min"]
    bounds = np.array(testfcts.testfunctions[function_name]["domain"])
    for i in range(n):
        sa = simulated_annealing(function_name, bounds, max_iterations, step_size, t0, tolerance)
        score = sa["best_eval"]
        it = sa["nb_iterations"]
        res = np.append(res,score)
        nb_it = np.append(nb_it,it)
    diff = abs(res-true_min) <= tolerance
    acc = np.count_nonzero(diff == True)/n
    nb_it = np.mean(nb_it)
    if nb_it == max_iterations:
        nb_it = "max d'itérations atteint"
    
    return({"fonction": function_name, "tolerance": tolerance,
            "accuracy": acc, "nb iterations moyen": nb_it})
    


rate_sa(100,"ackley",5000,0.6,10,1e-1)


rate_sa(500,"eggholder",1000,0.6,10,1e-1)



#min en fonction du nbr d'iterations du recuit

def vitesse_sa(n,function_name,max_iterations,step_size,t0,tolerance,ylim):
    res = np.zeros(max_iterations)
    bounds = np.array(testfcts.testfunctions[function_name]["domain"])
    true_min = testfcts.testfunctions[function_name]["global_min"]
    for i in range(n):
        sa = simulated_annealing(function_name, bounds, max_iterations, step_size, t0, tolerance)
        values = sa["values"]
        points = sa["points"]
        res = res + np.array(values)
    res = res/n
    x = list(range(max_iterations))
    
    tol_up = true_min+tolerance
    tol_low = true_min-tolerance
    
    plt.plot(x,res)
    plt.axhline(y=true_min, color='red', label="vrai min global")
    plt.axhline(y=tol_up,color='red',linestyle='--', linewidth=0.5, label="tolerance")
    plt.axhline(y=tol_low,color='red',linestyle='--', linewidth=0.5)
    plt.title("Vitesse de cvg du SA pour la fonction "+function_name)
    plt.xlabel("Itérations")
    plt.ylabel("Min")
    plt.legend()
    #plt.ylim(true_min-2, ylim)
    plt.show()
    
    #print(points)


vitesse_sa(100,"ackley",1000,0.6,100,1e-1,10)

vitesse_sa(100,"mccorm",1000,0.6,100,1e-1,10)


vitesse_sa(100,"himmelblau",1000,0.6,10,1e-1,10)



#fonction pour obtenir la liste de tous les min trouvés à chaque it en moyenne

def liste_min(n,function_name,max_iterations,step_size,t0,tolerance):
    res = np.zeros(max_iterations)
    bounds = np.array(testfcts.testfunctions[function_name]["domain"])
    for i in range(n):
        sa = simulated_annealing(function_name, bounds, max_iterations, step_size, t0, tolerance)
        values = sa["values"]
        res = res + np.array(values)
    res = res/n
    return(res)





    


#tx de succès de toutes les fcts tests en fonction d'une tolérance

def graph_rate(n,max_iterations,step_size,t0,tolerance): 
    res = {}
    for fct in testfcts.functions_names:
        rate = rate_sa(n,fct,max_iterations,step_size,t0,tolerance)
        res[fct] = rate["accuracy"]    

    rate = list(res.values())
    names = list(res.keys())
    
    fig, ax = plt.subplots()
    ax.barh(names, rate, align='center')
    #ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Taux')
    ax.set_title('Tx de succès pour tol '+str(tolerance)+",step_size= "+str(step_size)+", t0= "+str(t0))
    for i, v in enumerate(rate):
        ax.text(v + 0.01, i-0.1, str(v), color='blue', fontsize=8)
    
    plt.show()






graph_rate(100,5000,0.6,5000,1e-1)

graph_rate(500,1000,0.7,10,1e-1)


def iterations(n,max_iterations,step_size,t0,tolerance):
    res = {}
    for fct in testfcts.functions_names:
        rate = rate_sa(n,fct,max_iterations,step_size,t0,tolerance)
        res[fct] = rate["nb iterations moyen"]    
    return("Nb moyen d'iterations: ",res)


iterations(100,1000,0.6,100,1e-1)

    

#influence du step size



for k in (np.arange(0.1,1.0,0.1)):
    graph_rate(500,1000,k,10,1e-1)
    
    
    
    

for k in ([0.1,0.9]):
    res = liste_min(500,"camel",200,k,10,1e-1)
    x = list(range(200))
    plt.plot(x,res, label="step_size: "+str(k))
    plt.legend()


true_min = testfcts.testfunctions["camel"]["global_min"]

plt.axhline(y=true_min, color='red', label="vrai min global")
plt.axhline(y=true_min+0.1,color='red',linestyle='--', linewidth=0.5, label="tolerance")
plt.axhline(y=true_min-0.1,color='red',linestyle='--', linewidth=0.5)    
plt.xlabel("Itérations")
plt.ylabel("Min")
plt.title("Vitesse de cvg du SA pour la fct Camel")
plt.ylim(true_min-2,10)
plt.legend()
plt.show()





#influence de t0

for t0 in ([5,10,50,100,150,200]):
    graph_rate(500,1000,0.6,t0,1e-1)




t0_5 = iterations(500,1000,0.6,5,1e-1)
t0_10 = iterations(500,1000,0.6,10,1e-1)
t0_50 = iterations(500,1000,0.6,50,1e-1)
t0_100 = iterations(500,1000,0.6,100,1e-1)
t0_150 = iterations(500,1000,0.6,150,1e-1)



for fct in testfcts.functions_names:
    for t0 in ([5,10,50,100,150,200]):
        res = liste_min(500,fct,200,0.6,t0,1e-1)
        x = list(range(200))
        plt.plot(x,res, label="t0= "+str(t0))
        plt.legend()
    
    
    true_min = testfcts.testfunctions[fct]["global_min"]
    
    plt.axhline(y=true_min, color='red', label="vrai min global")
    plt.axhline(y=true_min+0.1,color='red',linestyle='--', linewidth=0.5, label="tolerance")
    plt.axhline(y=true_min-0.1,color='red',linestyle='--', linewidth=0.5)    
    plt.xlabel("Itérations")
    plt.ylabel("Min")
    plt.title("Vitesse de cvg du SA pour la fct "+fct)
    plt.ylim(true_min-2,2)
    plt.legend()
    plt.show()




################################################
########## COMPARAISON AVEC PACKAGE R ##########
################################################


import rpy2

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects



GenSA = importr("GenSA")

robjects.r('''
           
dimension <- 2
global.min <- -1.9133
tol <- 1e-09
lower <- c(-1.5,-3)
upper <- c(4,4)


#simulated annealing

out <- GenSA(lower = lower, upper = upper, fn = mccorm, control=list(threshold.stop=global.min+tol,verbose=TRUE))
out[c("value","par","counts")] ''')



mccorm = robjects.r('''
mccorm <- function(xx)
{
  x1 <- xx[1]
  x2 <- xx[2]
	
  term1 <- sin(x1 + x2)
  term2 <-(x1 - x2)^2
  term3 <- -1.5*x1
  term4 <- 2.5*x2
	
  y <- term1 + term2 + term3 + term4 + 1
  return(y)
}
''')


dimension = 2
global_min = -1.9133
tol = 1e-09
lower = robjects.r['c'](-1.5,-3)
upper = robjects.r['c'](4.0,4.0)

lower
upper

out = GenSA.GenSA(lower=lower, upper=upper, fn = mccorm)


