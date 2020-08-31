import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import genetic_functions as gen

# Without recursion

MAX_GENERATIONS = 10000

target = 100
elements = 6
population = 5
mutation_rate = 0.01
global_best = []
generation = 0


current_members = gen.generate_population(population, target, elements)
final_generation = 0
finished = False

for i in range(MAX_GENERATIONS):

    scores = []
    
    best1 = [np.array([0,0,0,0,0,0]),0.0]
    best2 = [np.array([0,0,0,0,0,0]),0.0]
        
    for gene in current_members:    
        score = gen.fitness(gene[0], target)
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
        finished = True
        final_generation = generation
        break

    parents = np.concatenate((best1[0],best2[0]))
    children = [best1, best2]    

    for i in range(elements - 2):
        child = gen.crossover(parents, elements)
        child = gen.mutate(child, mutation_rate)
        children.append([child, 0.0])     

    current_members = children
    generation += 1

if not finished:
    print('It didn\'t happen')

# Plot the evolution
# x = np.arange(0,final_generation + 1)
# y = np.array(global_best)
# plt.plot(x, y)
# plt.show()
