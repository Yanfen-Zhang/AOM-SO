"""
Adaptive-Oriented Mutation Snake Optimizer for Scheduling Budget-Constrained
Workflows in Heterogeneous Cloud Environments
"""

import math
import random

import numpy as np


class Task:

    def __init__(self, Pop_Size, Max_Iter, Dim, LB, UB, dag, comm_cost, calc_cost, cost_bgt_G, price):
        self.N = Pop_Size  # Population size
        self.T = Max_Iter  # Maximum iteration number
        self.dim = Dim  # Dimension
        self.lb = LB  # The minimum index of the available VMs
        self.ub = UB  # The maximum index of the available VMs
        self.dag = dag  # DAG
        self.comm_cost = comm_cost  # Communication time
        self.calc_cost = calc_cost  # Execution time
        self.cost_bgt_G = cost_bgt_G  # Budget
        self.price = price  # Unit price of the VMs

    def cost_value(self):
        """Calculate the execution cost of all tasks on all VMs separately."""
        cost_value = {}
        for ni in self.calc_cost:
            clist = []
            for vmk in range(len(self.price)):
                clist.append(self.calc_cost[ni][vmk] * self.price[vmk])
            cost_value[ni] = clist
        return cost_value

    def cost_min(self):
        """Calculate the minimum execution cost for each task individually."""
        cost_min = {}
        cost_value = self.cost_value()
        for ni in cost_value:
            cost_min[ni] = min(cost_value[ni])
        return cost_min

    def cost(self, p):
        """Calculate the total execution cost required by the scheduling solution represented by the individual."""
        cost = 0
        cost_value = self.cost_value()
        for i in range(len(p)):
            ni = i + 1
            vm = int(round(p[i]))
            cost = cost + cost_value[ni][vm]
        return cost

    def get_pred(self):
        """Obtain the predecessor task list for all tasks."""
        pre_dict = {}
        for ni in self.dag:
            pre_list = []
            for i in self.dag:
                data = []
                for j in self.dag[i]:
                    if j == ni:
                        data.append(i)
                        data.append(self.comm_cost[i][ni])
                        pre_list.append(data)
            pre_dict[ni] = pre_list
        return pre_dict

    def makespan(self, p):
        """Calculate the makespan required by the scheduling solution represented by the individual."""
        EST = {}
        EFT = {}
        VM = {}
        av = [0] * len(self.price)
        pre_dict = self.get_pred()
        for i in range(len(p)):
            ni = i + 1
            vm = int(round(p[i]))
            VM[ni] = vm
            pre_list = pre_dict[ni]
            if pre_list:
                ready = []
                for pre in pre_list:
                    if VM[pre[0]] == vm:
                        ready.append(EFT[pre[0]])
                    else:
                        ready.append(EFT[pre[0]] + pre[1])
                if max(ready) > av[vm]:
                    EST[ni] = max(ready)
                    EFT[ni] = EST[ni] + self.calc_cost[ni][vm]
                    av[vm] = EFT[ni]
                else:
                    EST[ni] = av[vm]
                    EFT[ni] = EST[ni] + self.calc_cost[ni][vm]
                    av[vm] = EFT[ni]
            else:
                EST[ni] = av[vm]
                EFT[ni] = EST[ni] + self.calc_cost[ni][vm]
                av[vm] = EFT[ni]
        makespan = max(EFT.values())
        return makespan

    def best(self, X, fitness_cost, fitness_makespan):
        """Find the best individual."""
        X_best = X[0]
        X_best_fitness_cost = fitness_cost[0]
        X_best_fitness_makespan = fitness_makespan[0]
        for i in range(1, len(X)):
            if fitness_cost[i] <= self.cost_bgt_G and X_best_fitness_cost <= self.cost_bgt_G:
                if fitness_makespan[i] < X_best_fitness_makespan:
                    X_best = X[i]
                    X_best_fitness_cost = fitness_cost[i]
                    X_best_fitness_makespan = fitness_makespan[i]
            elif fitness_cost[i] > self.cost_bgt_G and X_best_fitness_cost > self.cost_bgt_G:
                if fitness_cost[i] < X_best_fitness_cost:
                    X_best = X[i]
                    X_best_fitness_cost = fitness_cost[i]
                    X_best_fitness_makespan = fitness_makespan[i]
            else:
                if fitness_cost[i] <= self.cost_bgt_G and X_best_fitness_cost > self.cost_bgt_G:
                    X_best = X[i]
                    X_best_fitness_cost = fitness_cost[i]
                    X_best_fitness_makespan = fitness_makespan[i]
        return X_best, X_best_fitness_cost, X_best_fitness_makespan

    def worst(self, X, fitness_cost, fitness_makespan):
        """Find the worst individual."""
        X_worst_fitness_cost = fitness_cost[0]
        X_worst_fitness_makespan = fitness_makespan[0]
        worst_index = 0
        for i in range(1, len(X)):
            if fitness_cost[i] <= self.cost_bgt_G and X_worst_fitness_cost <= self.cost_bgt_G:
                if fitness_makespan[i] > X_worst_fitness_makespan:
                    worst_index = i
                    X_worst_fitness_cost = fitness_cost[i]
                    X_worst_fitness_makespan = fitness_makespan[i]
            elif fitness_cost[i] > self.cost_bgt_G and X_worst_fitness_cost > self.cost_bgt_G:
                if fitness_cost[i] > X_worst_fitness_cost:
                    worst_index = i
                    X_worst_fitness_cost = fitness_cost[i]
                    X_worst_fitness_makespan = fitness_makespan[i]
            else:
                if fitness_cost[i] > self.cost_bgt_G and X_worst_fitness_cost <= self.cost_bgt_G:
                    worst_index = i
                    X_worst_fitness_cost = fitness_cost[i]
                    X_worst_fitness_makespan = fitness_makespan[i]
        return worst_index

    def initial(self):
        """Generate initial population."""
        rows = self.N
        cols = self.dim
        r = np.random.rand(rows, cols)
        X = []  # Initial Population
        fitness_cost = []
        fitness_makespan = []
        for i in range(rows):
            p = []
            for j in range(cols):
                p.append((self.lb + r[i][j] * (self.ub - self.lb)))
            X.append(p)
            fitness_cost.append(self.cost(p))
            fitness_makespan.append(self.makespan(p))
        # Update the global best individual
        X_global_best, X_global_best_fitness_cost, X_global_best_fitness_makespan = self.best(X, fitness_cost,
                                                                                              fitness_makespan)
        return X, fitness_cost, fitness_makespan, X_global_best, X_global_best_fitness_cost, X_global_best_fitness_makespan

    def diving_groups(self):
        """Divide the initial population into male and female groups."""
        X, fitness_cost, fitness_makespan, X_global_best, X_global_best_fitness_cost, X_global_best_fitness_makespan = self.initial()
        Nm = int(round(self.N / 2))  # The number of individuals in the male group
        Nf = self.N - Nm  # The number of individuals in the female group
        Xm = X[:Nm]
        Xf = X[Nm:]
        # male
        fitness_cost_m = fitness_cost[:Nm]
        fitness_makespan_m = fitness_makespan[:Nm]
        # female
        fitness_cost_f = fitness_cost[Nm:]
        fitness_makespan_f = fitness_makespan[Nm:]
        # Update the best male individual
        X_m_best, X_m_best_fitness_cost, X_m_best_fitness_makespan = self.best(Xm, fitness_cost_m, fitness_makespan_m)
        # Update the best female individual
        X_f_best, X_f_best_fitness_cost, X_f_best_fitness_makespan = self.best(Xf, fitness_cost_f, fitness_makespan_f)
        return Nm, Nf, Xm, Xf, fitness_cost_m, fitness_makespan_m, fitness_cost_f, fitness_makespan_f, \
            X_global_best, X_global_best_fitness_cost, X_global_best_fitness_makespan, X_m_best, \
            X_m_best_fitness_cost, X_m_best_fitness_makespan, X_f_best, X_f_best_fitness_cost, X_f_best_fitness_makespan

    def vm_list(self, p, C):
        """Find the VM indexes that satisfy the task sub-budget."""
        slack_minus = C - self.cost_bgt_G
        cost_ni = {}
        cost_value = self.cost_value()
        cost_min = self.cost_min()
        cost_reduce = {}
        CAR = {}
        cost_CAR = {}
        vm_list = []
        for i in range(len(p)):
            ni = i + 1
            vm = int(round(p[i]))
            cost_ni[ni] = cost_value[ni][vm]
            cost_reduce[ni] = cost_ni[ni] - cost_min[ni]
        reduce_sum = sum(cost_reduce.values())
        for ni in self.dag:
            CAR[ni] = cost_reduce[ni] / reduce_sum
            cost_CAR[ni] = cost_ni[ni] - slack_minus * CAR[ni]
            av_vm = []
            for j in range(len(self.price)):
                if cost_value[ni][j] <= cost_CAR[ni]:
                    av_vm.append(j)
            if len(av_vm) == 0:
                av_vm.append(cost_value[ni].index(cost_min[ni]))
            vm_list.append(av_vm)
        return vm_list

    def iterate(self):
        """Starting the iterative optimisation process."""
        flag = [1, -1]  # # Random operators "+" or "-"
        QThreshold = 0.25  # Thresholds for the quantity of food
        TempThreshold = 0.6  # Threshold for temperature
        c1 = 0.5
        c2 = 0.05
        c3 = 2
        y = 0.1  # Oriented Probability
        y1 = int(round(self.dim * y))
        stage1 = []
        stage2 = []
        stage3 = []
        stage4 = []
        best_cost = []
        best_makespan = []
        Nm, Nf, Xm, Xf, fitness_cost_m, fitness_makespan_m, fitness_cost_f, fitness_makespan_f, \
            X_global_best, X_global_best_fitness_cost, X_global_best_fitness_makespan, X_m_best, \
            X_m_best_fitness_cost, X_m_best_fitness_makespan, X_f_best, X_f_best_fitness_cost, \
            X_f_best_fitness_makespan = self.diving_groups()
        for t in range(1, self.T+1):
            Temp = math.exp(-((t) / (self.T)))
            Q = c1 * math.exp(((t-self.T) / (self.T)))
            if Q > 1:
                Q = 1
            Xnewm = []
            Xnewf = []
            # Exploration Phase (no Food)
            if Q < QThreshold:
                stage1.append(t)
                # male
                for i in range(Nm):
                    new_list = []
                    for j in range(self.dim):
                        # random index：0 <= rand_leader_index <= Nm-1
                        arr = list(range(0, Nm))
                        rand_leader_index = random.choice(arr)
                        # positions of randomly selected male
                        Xm_rand = Xm[rand_leader_index]
                        # random operator "+" or "-"
                        flag_rand = random.choice(flag)
                        eps = np.finfo(float).eps
                        # exploration ability
                        Am = math.exp(-fitness_makespan_m[rand_leader_index] / (fitness_makespan_m[i] + eps))
                        new_list.append(Xm_rand[j] + flag_rand * c2 * Am * ((self.ub - self.lb) * random.random() + self.lb))
                    Xnewm.append(new_list)
                # female
                for i in range(Nf):
                    new_list = []
                    for j in range(self.dim):
                        # random index：0 <= rand_leader_index <= Nf-1
                        arr = list(range(0, Nf))
                        rand_leader_index = random.choice(arr)
                        # positions of randomly selected female
                        Xf_rand = Xf[rand_leader_index]
                        # random operator "+" or "-"
                        flag_rand = random.choice(flag)
                        eps = np.finfo(float).eps
                        # exploration ability
                        Af = math.exp(-fitness_makespan_f[rand_leader_index] / (fitness_makespan_f[i] + eps))
                        new_list.append(Xf_rand[j] + flag_rand * c2 * Af * (
                                    (self.ub - self.lb) * random.random() + self.lb))
                    Xnewf.append(new_list)
            # Exploitation Phase (Food Exists)
            else:
                # hot
                if Temp > TempThreshold:
                    stage2.append(t)
                    for i in range(Nm):
                        new_list = []
                        flag_rand = random.choice(flag)
                        for j in range(self.dim):
                            new_list.append(X_global_best[j] + flag_rand * c3 * Temp * random.random() * (X_global_best[j] - Xm[i][j]))
                        Xnewm.append(new_list)
                    for i in range(Nf):
                        new_list = []
                        flag_rand = random.choice(flag)
                        for j in range(self.dim):
                            new_list.append(X_global_best[j] + flag_rand * c3 * Temp * random.random() * (X_global_best[j] - Xf[i][j]))
                        Xnewf.append(new_list)
                # cold
                else:
                    # fight
                    if random.random() > 0.6:
                        stage3.append(t)
                        for i in range(Nm):
                            new_list = []
                            for j in range(self.dim):
                                eps = np.finfo(float).eps
                                FM = math.exp(-(X_f_best_fitness_makespan) / (fitness_makespan_m[i] + eps))
                                new_list.append(Xm[i][j] + c3 * FM * random.random() * (Q * X_f_best[j] - Xm[i][j]))
                            Xnewm.append(new_list)
                        for i in range(Nf):
                            new_list = []
                            for j in range(self.dim):
                                eps = np.finfo(float).eps
                                FF = math.exp(-(X_m_best_fitness_makespan) / (fitness_makespan_f[i] + eps))
                                new_list.append(Xf[i][j] + c3 * FF * random.random() * (Q * X_m_best[j] - Xf[i][j]))
                            Xnewf.append(new_list)
                    # mating
                    else:
                        stage4.append(t)
                        for i in range(Nm):
                            new_list = []
                            for j in range(self.dim):
                                eps = np.finfo(float).eps
                                Mm = math.exp(-fitness_makespan_f[i] / (fitness_makespan_m[i] + eps))
                                new_list.append(Xm[i][j] + c3 * Mm * random.random() * (
                                            Q * Xf[i][j] - Xm[i][j]))
                            Xnewm.append(new_list)
                        for i in range(Nf):
                            new_list = []
                            for j in range(self.dim):
                                eps = np.finfo(float).eps
                                Mf = math.exp(-fitness_makespan_m[i] / (fitness_makespan_f[i] + eps))
                                new_list.append(Xf[i][j] + c3 * Mf * random.random() * (
                                            Q * Xm[i][j] - Xf[i][j]))
                            Xnewf.append(new_list)
                        egg = random.choice(flag)
                        # Eggs hatch randomly
                        if egg == 1:
                            worstm = self.worst(Xm, fitness_cost_m, fitness_makespan_m)
                            worstf = self.worst(Xf, fitness_cost_f, fitness_makespan_f)
                            rm = self.lb + random.random() * (self.ub - self.lb)
                            rf = self.lb + random.random() * (self.ub - self.lb)
                            for w in range(self.dim):
                                Xnewm[worstm][w] = rm
                                Xnewf[worstf][w] = rf
            # male
            for i in range(Nm):
                for j in range(self.dim):
                    if Xnewm[i][j] > self.ub:
                        Xnewm[i][j] = self.ub
                    if Xnewm[i][j] < self.lb:
                        Xnewm[i][j] = self.lb
                # the superior individuals
                fitness_cost_new = self.cost(Xnewm[i])
                fitness_makespan_new = self.makespan(Xnewm[i])
                if fitness_cost_new <= self.cost_bgt_G and fitness_cost_m[i] <= self.cost_bgt_G:
                    if fitness_makespan_new < fitness_makespan_m[i]:
                        fitness_cost_m[i] = fitness_cost_new
                        fitness_makespan_m[i] = fitness_makespan_new
                        for j in range(self.dim):
                            Xm[i][j] = Xnewm[i][j]
                elif fitness_cost_new > self.cost_bgt_G and fitness_cost_m[i] > self.cost_bgt_G:
                    if fitness_cost_new < fitness_cost_m[i]:
                        fitness_cost_m[i] = fitness_cost_new
                        fitness_makespan_m[i] = fitness_makespan_new
                        for j in range(self.dim):
                            Xm[i][j] = Xnewm[i][j]
                else:
                    if fitness_cost_new <= self.cost_bgt_G and fitness_cost_m[i] > self.cost_bgt_G:
                        fitness_cost_m[i] = fitness_cost_new
                        fitness_makespan_m[i] = fitness_makespan_new
                        for j in range(self.dim):
                            Xm[i][j] = Xnewm[i][j]
            # female
            for i in range(Nf):
                for j in range(self.dim):
                    if Xnewf[i][j] > self.ub:
                        Xnewf[i][j] = self.ub
                    if Xnewf[i][j] < self.lb:
                        Xnewf[i][j] = self.lb
                # the superior individuals
                fitness_cost_new = self.cost(Xnewf[i])
                fitness_makespan_new = self.makespan(Xnewf[i])
                if fitness_cost_new <= self.cost_bgt_G and fitness_cost_f[i] <= self.cost_bgt_G:
                    if fitness_makespan_new < fitness_makespan_f[i]:
                        fitness_cost_f[i] = fitness_cost_new
                        fitness_makespan_f[i] = fitness_makespan_new
                        for j in range(self.dim):
                            Xf[i][j] = Xnewf[i][j]
                elif fitness_cost_new > self.cost_bgt_G and fitness_cost_f[i] > self.cost_bgt_G:
                    if fitness_cost_new < fitness_cost_f[i]:
                        fitness_cost_f[i] = fitness_cost_new
                        fitness_makespan_f[i] = fitness_makespan_new
                        for j in range(self.dim):
                            Xf[i][j] = Xnewf[i][j]
                else:
                    if fitness_cost_new <= self.cost_bgt_G and fitness_cost_f[i] > self.cost_bgt_G:
                        fitness_cost_f[i] = fitness_cost_new
                        fitness_makespan_f[i] = fitness_makespan_new
                        for j in range(self.dim):
                            Xf[i][j] = Xnewf[i][j]

            # Adaptive-Oriented Mutation

            # male
            rand_list = list(range(0, Nm))
            u = int(round(Nm * y))
            p_rand_list = random.sample(rand_list, u)
            for p_rand in p_rand_list:
                if fitness_cost_m[p_rand] > self.cost_bgt_G:
                    vm_list = self.vm_list(Xm[p_rand], fitness_cost_m[p_rand])
                    j_list = random.sample(list(range(0, self.dim)), y1)
                    for j in j_list:
                        Xm[p_rand][j] = random.choice(vm_list[j])
                    fitness_cost_m[p_rand] = self.cost(Xm[p_rand])
                    fitness_makespan_m[p_rand] = self.makespan(Xm[p_rand])
            # female
            rand_list = list(range(0, Nf))
            u = int(round(Nf * y))
            p_rand_list = random.sample(rand_list, u)
            for p_rand in p_rand_list:
                if fitness_cost_f[p_rand] > self.cost_bgt_G:
                    vm_list = self.vm_list(Xf[p_rand], fitness_cost_f[p_rand])
                    j_list = random.sample(list(range(0, self.dim)), y1)
                    for j in j_list:
                        Xf[p_rand][j] = random.choice(vm_list[j])
                    fitness_cost_f[p_rand] = self.cost(Xf[p_rand])
                    fitness_makespan_f[p_rand] = self.makespan(Xf[p_rand])

            # Update the best male individual
            new_best_m, new_best_m_fitness_cost, new_best_m_fitness_makespan = self.best(Xm, fitness_cost_m,
                                                                                         fitness_makespan_m)
            X_list_m = [new_best_m, X_m_best]
            cost_list_m = [new_best_m_fitness_cost, X_m_best_fitness_cost]
            makespan_list_m = [new_best_m_fitness_makespan, X_m_best_fitness_makespan]
            X_m_best, X_m_best_fitness_cost, X_m_best_fitness_makespan = self.best(X_list_m, cost_list_m, makespan_list_m)

            # Update the best female individual
            new_best_f, new_best_f_fitness_cost, new_best_f_fitness_makespan = self.best(Xf, fitness_cost_f,
                                                                                         fitness_makespan_f)
            X_list_f = [new_best_f, X_f_best]
            cost_list_f = [new_best_f_fitness_cost, X_f_best_fitness_cost]
            makespan_list_f = [new_best_f_fitness_makespan, X_f_best_fitness_makespan]
            X_f_best, X_f_best_fitness_cost, X_f_best_fitness_makespan = self.best(X_list_f, cost_list_f, makespan_list_f)
            # Update the best individual
            list1 = [X_m_best, X_f_best]
            list2 = [X_m_best_fitness_cost, X_f_best_fitness_cost]
            list3 = [X_m_best_fitness_makespan, X_f_best_fitness_makespan]
            X_global_best, X_global_best_fitness_cost, X_global_best_fitness_makespan = self.best(list1, list2, list3)
            best_cost.append(X_global_best_fitness_cost)
            best_makespan.append(X_global_best_fitness_makespan)
        return X_global_best, X_global_best_fitness_cost, X_global_best_fitness_makespan, best_cost, best_makespan

    def results(self):
        X_global_best, X_global_best_cost, X_global_best_makespan, best_cost, best_makespan = self.iterate()
        cost_min_G = sum(self.cost_min().values())
        return X_global_best, X_global_best_cost, X_global_best_makespan, best_cost, best_makespan, cost_min_G
