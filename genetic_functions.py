import numpy as np
import numpy.ma as ma
def fitness(gene, target):
    sum_ = gene.sum()
    if sum_ > target:
        return pow(target / sum_, 2)
    return pow(sum_ / target, 2)

def generate_population(population, target, elements):
    gen = []
    for i in range(population):
        g = np.random.randint(1,target,elements)
        gen.append([g, 0.0])
    return gen

def crossover(parents, elements):    
    y1 = np.zeros((elements), np.uint8)
    y2 = np.ones((elements), np.uint8)
    y = np.concatenate((y1,y2))
    np.random.shuffle(y)        
    child = ma.masked_array(parents, mask=y).compressed()
    a = np.unique(child, return_inverse=True)
    print(a)
    return child

def mutate(child,mutation_rate = 0.01):            
    for i, g in enumerate(child):            
        if mutation_rate > np.random.rand():
            print('Mutation has ocurred')
            child[i] = np.random.randint(1, 100)
    return child