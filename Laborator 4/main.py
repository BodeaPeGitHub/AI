from random import seed

from chromosomes.SPChromosome import SPChromosome
from chromosomes.TSPChromosome import TSPChromosome
from genetic_algorithm import GA
from plot_network import plot_network
from read_graf import from_matrix_to_network, from_matrix_to_network_with_semicolon, from_tsp_to_network
import warnings

warnings.simplefilter('ignore')


def calculate_path(path, problem_parameters):
    cost = 0
    matrix = problem_parameters['matrix']
    for x, y in zip(path[:-1], path[1:]):
        cost += matrix[x][y]
    return cost


def genetic_algoritm(algorithm_parameters, problem_parameters, chromosome):
    ga = GA(algorithm_parameters, problem_parameters, chromosome)
    same = None
    same_count = 0
    ga.initialisation()
    ga.evaluation()
    generation = 0
    while generation < algorithm_parameters['generations'] and same_count < 50:
        # ga.one_generation()
        # ga.one_generation_elitism()
        ga.one_generation_steady_state()
        ch = ga.best_chromosome()
        if same == ch:
            same_count += 1
        else:
            same = ch
            same_count = 1
        print(generation, ch)
        generation += 1

    bests = ga.find_all_best_chromosomes()
    return bests


def solve_problem(algorithm_parameters, problem_parameters, name, chromosome, file):
    bests = genetic_algoritm(algorithm_parameters, problem_parameters, chromosome)
    solution = {'problem_name': name, 'no_of_cities': len(bests[0].representation), 'paths': [], 'finesses': 0}
    for solutions in bests:
        representation = [i + 1 for i in solutions.representation]
        fitness = solutions.fitness
        solution['paths'].append(representation)
        solution['finesses'] = fitness
    name += '\n' + str(solution['paths'][0]) + '\n' + 'Fitness: ' + str(solution['finesses']) + '.'
    plot_network(problem_parameters, name=name, path=solution['paths'][0])
    file_text = [str(key) + ': ' + str(value) + '\n' for key, value in solution.items()]
    file.writelines(file_text)
    return solution

if __name__ == '__main__':
    path = 'data/easy/example-easy-not-full.txt'
    path = 'data/eil51.txt'
    problem_parameters = from_tsp_to_network(path)
    problem_parameters['function'] = calculate_path
    algorithm_parameters = {
        'popSize': 500,
        'generations': 800,
    }
    with open('data/solutions/solution-' + path.split('/')[-1], 'w+') as file:
        tsp_sol = solve_problem(algorithm_parameters, problem_parameters, 'TSP Problem', TSPChromosome, file)
        # sp_sol = solve_problem(algorithm_parameters, problem_parameters, 'SP Problem', SPChromosome, file)
    print(tsp_sol)
    # print(sp_sol)
