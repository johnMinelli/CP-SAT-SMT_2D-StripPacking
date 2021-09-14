from itertools import combinations

from z3 import *
import matplotlib.pyplot as plt

import time
import fileinput


def read_file_instance(n_instance):
    s = ''
    filepath = "../../instances/ins-{}.txt".format(n_instance)
    for line in fileinput.input(files=filepath):
        s += line
    return s.splitlines()


def display_solution(title, sizes_plate, n_circuits, sizes_circuits, pos_circuits):
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('hsv', n_circuits)
    ax = plt.gca()
    plt.title(title)
    if len(pos_circuits) > 0:
        for i in range(n_circuits):
            rect = plt.Rectangle(pos_circuits[i], *sizes_circuits[i], edgecolor="#333", facecolor=cmap(i))
            ax.add_patch(rect)
    ax.set_xlim(0, sizes_plate[0])
    ax.set_ylim(0, sizes_plate[1] + 1)
    ax.set_xticks(range(sizes_plate[0] + 1))
    ax.set_yticks(range(sizes_plate[1] + 1))
    ax.set_xlabel('width_plate')
    ax.set_ylabel('height_plate')

    plt.show()


def write_file_output(n_instance, sizes_plate, n_circuits, sizes_circuits, pos_circuits):
    filepath = "../out/out−{}.txt".format(n_instance)
    f = open(filepath, "w+")
    f.write("{} {}\n".format(*sizes_plate))
    f.write("{}\n".format(n_circuits))
    for i in range(len(pos_circuits)):
        f.write("{} {} {} {}\n".format(*sizes_circuits[i], *pos_circuits[i]))
    f.close()


def format_solution(solution):
    if not solution is None:
        X = solution["X"]
        Y = solution["Y"]
        positions = [(x, y) for x, y in zip(X, Y)]
        height = solution["HEIGHT"]
        status = solution["status"]
    else:
        positions = []
        height = 0
        status = "FAILED"
    conflicts = solution["conflicts"]
    return positions, height, {"time": time, "conflicts": conflicts, "status": status }


def evaluate_instance(n_instance):
    problem_instance = read_file_instance(n_instance)
    # Parameters
    width_plate = int(problem_instance[0])
    n_circuits = int(problem_instance[1])
    sizes_circuits = [[int(val) for val in i.split()] for i in problem_instance[-n_circuits:]]
    input_order, sorted_sizes_circuits = zip(*[(index, value) for index, value in
                                               sorted(enumerate(sizes_circuits), reverse=True,
                                                      key=lambda x: x[1][0] * x[1][1])])
    params_instance = dict()
    params_instance["width"] = width_plate
    params_instance["n_rects"] = n_circuits
    params_instance["width_rects"] = [sizes[0] for sizes in sorted_sizes_circuits]
    params_instance["height_rects"] = [sizes[1] for sizes in sorted_sizes_circuits]

    opt = Optimize()
    opt.set(timeout=300000)  # in milliseconds

    start = time.time()
    positions, height_plate, stats = format_solution(solve_2DSP(opt, params_instance))
    exec_time = "{:.2f}".format(time.time()-start)

    # print("Instance solved in {}".format(end - start))

    # restore the original order for the results
    sizes_plate = (width_plate, height_plate)
    original_order_positions = [x for x, _ in sorted(zip(positions, input_order), key=lambda x: x[1])]
    display_solution("n°{} - Solved in {}".format(n_instance, exec_time), sizes_plate, n_circuits, sizes_circuits,
                     original_order_positions)
    write_file_output(n_instance, sizes_plate, n_circuits, sizes_circuits, original_order_positions)
    print(exec_time, stats["conflicts"], "      ", stats["status"])

# We have variables for position for each circuit with coords from bottom right corner
# X_i is the coordinate on the horizontal axis of ri
# Y_i is the coordinate on the vertical exis of ri

# Them are difined with the following domain values 
# D(xi) = {a ∊ N | 0 <= a <= W - wi}
# D(yi) = {a ∊ N | 0 <= a <= H - hi}

# each rectangle is represented in relation to the others: n^2 variables for each axis
# to represent the position of a rectangle i wrt j are used lr_i,j ud_i,j
# lr_i,j is true if ri is placed at the left to the rj.
# ud_i,j is true if ri is placed at the downward to the rj.


def at_least_one(bool_vars):
    return Or(bool_vars)

def at_most_one(bool_vars):
    return [Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)]

def exactly_one(solver, bool_vars):
    solver.add(at_most_one(bool_vars))
    solver.add(at_least_one(bool_vars))

def iff(solver, lhs, rhs):
    solver.add(And(Or(Not(lhs), rhs), Or(Not(rhs), lhs)))

def positive_range(b):
    if b < 0: return []
    return range(0, b)


def indomain(box_size, rect_size):
    return box_size - rect_size >= 0


def solve_2DSP(optimizer, params):
    # init the variables from the params
    # width of bounding box (= circuit plate).
    W = params["width"]
    # number of rectangles (= circuits).
    n_rects = params["n_rects"]
    # widths of rectangles
    width_rects = params["width_rects"]
    # heights of rectangles
    height_rects = params["height_rects"]
    # bounds
    min_h = max(height_rects)
    max_h = sum(height_rects)

    # Variables for 2DSP
    H = Int("h")
    lr = [[Bool(f"lr_{i}_{j}") for j in range(n_rects)] for i in range(n_rects)]
    ud = [[Bool(f"ud_{i}_{j}") for j in range(n_rects)] for i in range(n_rects)]
    X = [Int(f"x_{i}") for i in range(n_rects)]
    Y = [Int(f"y_{i}") for i in range(n_rects)]

    # domain encoding for H
    optimizer.add([min_h <= H, H <= max_h])
    # domain encoding for X and Y
    for i in range(n_rects):
        optimizer.add([0 <= X[i], X[i] <= W-width_rects[i]])
        optimizer.add([0 <= Y[i], Y[i] <= H-height_rects[i]])

    # non overlapping constraint (2)(3)
    for i in range(n_rects):
        for j in range(n_rects):
            if i < j:
                # domain constraint for X and Y in relation to lr and ud
                optimizer.add(Implies(lr[i][j], X[j]>=width_rects[i]))
                optimizer.add(Implies(lr[j][i], X[i]>=width_rects[j]))
                optimizer.add(Implies(ud[i][j], Y[j]>=height_rects[i]))
                optimizer.add(Implies(ud[j][i], Y[i]>=height_rects[j]))

                # # symmetries for the problem
                if (width_rects[i], height_rects[i]) == (width_rects[j], height_rects[j]):
                    # SR
                    optimizer.add(Or(lr[i][j], ud[i][j], ud[j][i]))
                    optimizer.add(Implies(ud[i][j], lr[j][i]))

                    optimizer.add(Implies(lr[i][j], X[i] + width_rects[i] <= X[j]))
                    optimizer.add(Implies(ud[i][j], Y[i] + height_rects[i] <= Y[j]))
                    optimizer.add(Implies(ud[j][i], Y[j] + height_rects[j] <= Y[i]))
                # elif width_rects[i]+width_rects[j]>W:
                #     # LRH (only h, this technique in the vertical direction doesn't perform good)
                #     optimizer.add(Or(ud[i][j], ud[j][i]))
                # 
                #     optimizer.add(Implies(ud[i][j], Y[i] + height_rects[i] <= Y[j]))
                #     optimizer.add(Implies(ud[j][i], Y[j] + height_rects[j] <= Y[i]))
                # elif width_rects[i]>(W-width_rects[j])/2:
                #     # LS domain reduction 
                #     optimizer.add(Or(lr[j][i], ud[i][j], ud[j][i]))
                # 
                #     optimizer.add(Implies(lr[j][i], X[j] + width_rects[j] <= X[i]))
                #     optimizer.add(Implies(ud[i][j], Y[i] + height_rects[i] <= Y[j]))
                #     optimizer.add(Implies(ud[j][i], Y[j] + height_rects[j] <= Y[i]))
                else:
                    # regular case
                    optimizer.add(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))
                    optimizer.add(Implies(lr[i][j], X[i] + width_rects[i] <= X[j]))
                    optimizer.add(Implies(lr[j][i], X[j] + width_rects[j] <= X[i]))
                    optimizer.add(Implies(ud[i][j], Y[i] + height_rects[i] <= Y[j]))
                    optimizer.add(Implies(ud[j][i], Y[j] + height_rects[j] <= Y[i]))

    optimizer.minimize(H)

    min_H = 0
    pos_X = []
    pos_Y = []
    if optimizer.check() == sat:
        model = optimizer.model()
        min_H = int(model.evaluate(H).as_string())
        # print(f"Found best height at {min_H}")

        for i in range(n_rects):
            pos_X.append(int(model.evaluate(X[i]).as_string()))
            pos_Y.append(int(model.evaluate(Y[i]).as_string()))
        status = "SOLVED"
    elif optimizer.reason_unknown() == "timeout":
        # print("Solver timeout")
        status = "TIMEOUT"
    else:
        # print("Failed to solve")
        status = "FAILED"
    return {"X": pos_X, "Y": pos_Y, "HEIGHT": min_H, "status": status,
            "conflicts": optimizer.statistics().conflicts}


if __name__ == '__main__':
    N_INSTANCE = 10

    # for i in range(1, 41):
    #     evaluate_instance(i)

    evaluate_instance(N_INSTANCE)