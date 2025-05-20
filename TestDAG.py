import AOM_SO

# DAG
dag = {1: (2, 3, 4, 5, 6), 2: (8, 9), 3: (7,), 4: (8, 9), 5: (9,),
       6: (8,), 7: (10,), 8: (10,), 9: (10,), 10: ()}

# Communication time
comm_cost = {
    1: {2: 10, 3: 9, 4: 13, 5: 17, 6: 12},
    2: {8: 17, 9: 4},
    3: {7: 15},
    4: {8: 23, 9: 8},
    5: {9: 17},
    6: {8: 26},
    7: {10: 21},
    8: {10: 9},
    9: {10: 17},
    10: {}
}

# Execution time
calc_cost = {1: (13, 10, 7), 2: (19, 14, 10), 3: (17, 13, 9), 4: (21, 16, 12), 5: (14, 10, 8),
             6: (20, 15, 11), 7: (22, 17, 13), 8: (18, 13, 10), 9: (23, 18, 13), 10: (15, 11, 8)}

# Budget
cost_bgt_G = 650

# Unit price of the VMs
price = (3, 5, 7)


Pop_Size = 50  # Population size
Max_Iter = 1000  # Maximum iteration number
Dim = len(dag)  # Dimension
LB = 0  # The minimum index of the available VMs
UB = len(price)-1  # The maximum index of the available VMs


TestObj = AOM_SO.Task(Pop_Size, Max_Iter, Dim, LB, UB, dag, comm_cost, calc_cost, cost_bgt_G, price)
X_global_best, X_global_best_cost, X_global_best_makespan, best_cost, best_makespan, cost_min_G = TestObj.results()

print(f"The minimum cost required to execute workflow: {cost_min_G}")
print(f"Budget constraint：{cost_bgt_G}")
print(f"Total execution cost: {X_global_best_cost}")
print(f"Makespan: {X_global_best_makespan}")



