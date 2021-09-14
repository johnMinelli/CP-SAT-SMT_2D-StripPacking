import fileinput
import time

from datetime import timedelta

from matplotlib import pyplot as plt
from minizinc import Instance, Model, Solver

def read_file_instance(n_instance):
    s = ''
    filepath =  "../../instances/ins-{}.txt".format(n_instance)
    for line in fileinput.input(files = filepath):
        s += line
    return s.splitlines()

def write_file_output(n_instance, sizes_plate, n_circuits, sizes_circuits, pos_circuits):
    filepath = "../out/out−{}.txt".format(n_instance)
    f = open(filepath, "w")
    f.write("{} {}\n".format(*sizes_plate))
    f.write("{}\n".format(n_circuits))
    for i in range(len(pos_circuits)):
        f.write("{} {} {} {}\n".format(*sizes_circuits[i], *pos_circuits[i]))
    f.close()


def format_solution(solution):
    if not solution.solution is None:
        X = solution.solution.X
        Y = solution.solution.Y
        positions = [(x,y) for x,y in zip(X, Y)]
        height = solution.solution.HEIGHT
    else:
        positions = []
        height = 0
    time = str(solution.statistics["time"])
    sol = solution.statistics["solutions"]
    fails = solution.statistics["failures"]
    status = str(solution.status)

    return positions, height, {"time": time, "sol": sol, "fails": fails, "status": status}

def display_solution(title, sizes_plate, n_circuits, sizes_circuits, pos_circuits):
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('hsv', n_circuits)
    ax = plt.gca()
    plt.title(title)
    if len(pos_circuits)>0:
        for i in range(n_circuits):
            rect = plt.Rectangle(pos_circuits[i], *sizes_circuits[i], edgecolor="#333", facecolor=cmap(i))
            ax.add_patch(rect)
    ax.set_xlim(0, sizes_plate[0])
    ax.set_ylim(0, sizes_plate[1]+1)
    ax.set_xticks(range(sizes_plate[0] + 1))
    ax.set_yticks(range(sizes_plate[1] + 1))
    ax.set_xlabel('width_plate')
    ax.set_ylabel('height_plate')

    plt.show()


def evaluate_instance(model_instance, n_instance):

    problem_instance = read_file_instance(n_instance)
    # Parameters
    width_plate = int(problem_instance[0])
    n_circuits = int(problem_instance[1])
    sizes_circuits = [[int(val) for val in i.split()] for i in problem_instance[-n_circuits:]]
    input_order, sorted_sizes_circuits  = zip(*[(index, value) for index, value in 
                                   sorted(enumerate(sizes_circuits), reverse=True, key=lambda x: x[1][0] * x[1][1])])

    model_instance["width"] = width_plate
    model_instance["n_rects"] = n_circuits
    model_instance["width_rects"] = [sizes[0] for sizes in sorted_sizes_circuits]
    model_instance["height_rects"] = [sizes[1] for sizes in sorted_sizes_circuits]

    start = time.time()
    positions, height_plate, minizinc_stats = format_solution(model_instance.solve(timeout=timedelta(seconds=300), processes=8))
    end = time.time()

    # print("Instance solved in {}".format(minizinc_stats["time"]))
    # print("H: {};  POS: {}".format(positions, height_plate))

    # restore the original order for the results
    sizes_plate = (width_plate, height_plate)
    original_order_positions = [x for x, _ in sorted(zip(positions, input_order), key=lambda x: x[1])]
    display_solution("n°{} - Solved in {}, {}/{}".format(n_instance, minizinc_stats["time"],minizinc_stats["sol"],minizinc_stats["fails"]), sizes_plate, n_circuits, sizes_circuits, original_order_positions)
    write_file_output(N_INSTANCE, sizes_plate, n_circuits, sizes_circuits, original_order_positions)
    print(minizinc_stats["time"],minizinc_stats["sol"],minizinc_stats["fails"], "      ", minizinc_stats["status"])


if __name__ == '__main__':

    N_INSTANCE = 10
    # Load model from file
    model = Model("./full_model.mzn")
    # Find the MiniZinc solver configuration for Gecode
    gecode = Solver.lookup("gecode")
    # Create an Instance of the model for the solver
    # ---
    # for i in range(1,41):
    #     model_instance = Instance(gecode, model)
    #     evaluate_instance(model_instance, i)
    # ---
    model_instance = Instance(gecode, model)

    evaluate_instance(model_instance, N_INSTANCE)