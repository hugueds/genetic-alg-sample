import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

MAX_GENERATIONS = 999

target = 100
elements = 6
population = 5
mutation_rate = 0.01
global_best = []
generation = 0

y1 = np.zeros((elements), np.uint8)
y2 = np.ones((elements), np.uint8)
y = np.concatenate((y1,y2))

def fitness(gene, target):
    sum_ = gene.sum()
    if sum_ > target:
        return target / sum_
    return sum_ / target

def generate_population(population, target, elements):
    gen = []
    for i in range(population):
        g = np.random.randint(1,target,elements)
        gen.append([g, 0.0])
    return gen

def crossover():
    pass

def mutate():
    pass 

def run(generation, current_members):

    global MAX_GENERATIONS, mutation_rate, global_best, y    

    if generation > MAX_GENERATIONS:
        print(current_members[0][0], current_members[0][0].sum())
        return generation
    
    scores = []
    
    best1 = [np.array([0,0,0,0,0,0]),0.0]
    best2 = [np.array([0,0,0,0,0,0]),0.0]

    for gene in current_members:    
        score = fitness(gene[0], target)
        if score > best1[1]:
            best1 = [gene[0], score]
        elif score > best2[1]:
            best2 = [gene[0], score]
        scores.append(score)

    arr_scores = np.array(scores)
    print(arr_scores.mean())
    global_best.append(best1[1])

    if best1[1] == 1.0:
        print(best1[0])
        print(f'You made it in gen: {generation}')
        return generation

    children = [best1, best2]    

    c = np.concatenate((best1[0],best2[0]))

    for i in range(elements - 2):        
        np.random.shuffle(y)        
        child = ma.masked_array(c, mask=y).compressed()        
        for i, g in enumerate(child):            
            if mutation_rate > np.random.rand():
                print('Mutation has ocurred')
                child[i] = np.random.randint(1, 100)
        children.append([child, 0.0])

    current_members = children
    generation += 1    
    return run(generation, current_members)

current_members = generate_population(population, target, elements)

final_gen = run(generation, current_members)

 


x = np.arange(0,final_gen + 1)
y_ = np.array(global_best)
plt.plot(x, y_)
plt.show()
    