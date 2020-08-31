import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# WITH RECURSION

MAX_GENERATIONS = 999

target = 100
elements = 6
population = 5
mutation_rate = 0.01
global_best = []
generation = 0


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

def crossover(parents):
    global elements
    y1 = np.zeros((elements), np.uint8)
    y2 = np.ones((elements), np.uint8)
    y = np.concatenate((y1,y2))
    np.random.shuffle(y)        
    child = ma.masked_array(parents, mask=y).compressed()
    return child

def mutate(child):            
    for i, g in enumerate(child):            
        if mutation_rate > np.random.rand():
            print('Mutation has ocurred')
            child[i] = np.random.randint(1, 100)
    return child

def run(generation, current_members):

    global MAX_GENERATIONS, mutation_rate, global_best, y    

    if generation > MAX_GENERATIONS:
        print(f'Max generation exceeded')
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

    print(f'Generation Mean: {np.array(scores).mean()}')
    global_best.append(best1[1])

    if best1[1] == 1.0:
        print(f'You made it in gen: {generation}')
        print(best1[0])
        return generation

    parents = np.concatenate((best1[0],best2[0]))
    children = [best1, best2]    

    for i in range(elements - 2):
        child = crossover(parents)
        child = mutate(child)
        children.append([child, 0.0])     

    current_members = children
    generation += 1

    return run(generation, current_members)

current_members = generate_population(population, target, elements)

final_gen = run(generation, current_members) 

# Plot the evolution
x = np.arange(0,final_gen + 1)
y = np.array(global_best)
plt.plot(x, y)
plt.show()
