#from min_J_2 import K
from math import inf
from mmap import MADV_DOFORK
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import _select_dispatcher
from sympy import *
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr
from sympy.parsing.sympy_parser import standard_transformations
import fuzzy_reg
import numpy as np
import gen_alg
import random
import copy

import time

class Integrator:
        def __init__(self, init_val):
                self.value = init_val
        
        def integrate_step(self, input, step):
                self.value = self.value + input * step

class MyDerivative:
        def __init__(self, init_input):
                self.value = 0
                self.prev_input = init_input
        
        def deriv_step(self, input, step):
                self.value = (input - self.prev_input) / step
                self.prev_input = input

class Runge_Kutta:
        
        def __init__(self, a_args, b_args, init_vals, fx, h, x_limit, M_kp, M_ki, M_kd, Tr):
                self.a_args = [x / a_args[0] for x in a_args]
                self.b_args = [0] * (len(a_args) - len(b_args))
                self.b_args.extend([x / a_args[0] for x in b_args])
                self.k_args = [self.b_args[0], ]
                for i in range(1, len(self.a_args)):
                        tmp = self.b_args[i]
                        for j in range(1, i):
                                tmp -= self.a_args[j] * self.k_args[i - j]
                        self.k_args.append(tmp)
       
                self.init_vals = copy.deepcopy(init_vals)
                self.fx = str(fx)
                self.h = h
                self.x_limit = x_limit
                self.z0_solve = list()
                self.z1_solve = list()

                # Error lists for quadratic perfomance metric
                self.e0_solve = list()
                self.e1_solve = list()

                self.M_kp = copy.deepcopy(M_kp)
                self.M_kd = copy.deepcopy(M_kd)
                self.M_ki = copy.deepcopy(M_ki)
                self.Tr = Tr
                  
                self.x = symbols('x')
                transformations = (standard_transformations + (implicit_multiplication_application,))
                self.fx = parse_expr(self.fx, transformations=transformations)
                self.e = 1
                self.J_list = list()

        def f_x(self, f, X):
                return f.subs(self.x, X)

        def  k_for_other_z(self, u, z, k, h, index):
                if index == 0:
                        for i in range(len(z)-1):
                                k[index, i] = z[i+1] + u * self.k_args[i + 1]
                elif index == 3:
                        for i in range(len(z)-1):
                                k[index, i] = z[i+1] + k[index-1, i+1] * h + u * self.k_args[i+1]
                else:
                        for i in range(len(z)-1):
                                k[index, i] = z[i+1] + k[index-1, i+1] * h/2 + u * self.k_args[i+1]

        def k_for_z_minus_one(self, u, z, k, h, index):
                k[index, -1] = u * self.k_args[-1]
                if index == 0:
                        for i in range(len(z)):
                                k[index, -1] -= self.a_args[len(z) - i]*z[i]
                elif index == 3:
                        for i in range(len(z)):
                                k[index, -1] -= self.a_args[len(z) - i]*(z[i] + k[index-1, i] * h)
                else:
                        for i in range(len(z)):
                                k[index, -1] -= self.a_args[len(z) - i]*(z[i] + k[index-1, i] * h/2)

        def plot_e(self):
                X = np.linspace(0, self.x_limit, len(self.e_list))
                plt.plot(X, self.e_list)
                plt.grid()
                plt.show()

        def solve(self):
                self.z0_solve = [self.init_vals[0], ]
                z_vals = self.init_vals[:]
                X = 0

                index = int(self.x_limit/self.h)
                coef_k = np.zeros((4, len(z_vals)))

                intgr = Integrator(0)
                drv = MyDerivative(self.f_x(self.fx, X) - self.z0_solve[-1])
                
                for i in range(index):
                        
                        # 1. Calculate error of system control signal value
                        self.e = self.f_x(self.fx, X) - self.z0_solve[-1]
                        #print('e = {}'.format(self.e))

                        # 2. Calculate control signal value
                        U = 0
                        if self.M_kp is not None:
                                tmp = fuzzy_reg.fuzzy_reg_K(self.M_kp, self.e)
                                U += tmp.calc_out()
                        if self.M_ki is not None:
                                tmp = fuzzy_reg.fuzzy_reg_K(self.M_ki, intgr.value)
                                U += tmp.calc_out()
                        if self.M_kd is not None:
                                tmp = fuzzy_reg.fuzzy_reg_K(self.M_kd, drv.value)
                                U += tmp.calc_out()
                        
                         #вычисляем К1 для z0, z1, ... , zn-1
                        self.k_for_z_minus_one(U, z_vals, coef_k, self.h, index = 0)
                        self.k_for_other_z(U, z_vals, coef_k, self.h, index=0)

                        self.k_for_z_minus_one(U, z_vals, coef_k, self.h, index = 1)
                        self.k_for_other_z(U, z_vals, coef_k, self.h, index=1)

                        self.k_for_z_minus_one(U, z_vals, coef_k, self.h, index = 2)
                        self.k_for_other_z(U, z_vals, coef_k, self.h, index=2)

                        self.k_for_z_minus_one(U, z_vals, coef_k, self.h, index = 3)
                        self.k_for_other_z(U, z_vals, coef_k, self.h, index=3)

                        X+=self.h
                        #вычисляем новое значение zn-1
                        z_vals[-1] += (coef_k[0, -1] + 2*coef_k[1, -1] + \
                                2*coef_k[2, -1] + coef_k[3, -1]) * self.h/6
                        #вычисляем новые значения для z0, z1, ... , zn-2
                        for i in range(len(z_vals)-1):
                                z_vals[i] += (coef_k[0, i] + 2*coef_k[1, i] \
                                        + 2*coef_k[2, i] + coef_k[3, i]) * self.h/6
                        
                        
                        self.z1_solve.append(z_vals[1])
                        self.z0_solve.append(z_vals[0] + self.k_args[0] * U)


                        tmp = (self.f_x(self.fx, X) - self.z0_solve[-1])
                        self.e0_solve.append(tmp)

                        # WARNING ONLY WORK FOR CONSTANT VALUE. IN ANY OTHER INPUT TYPE CAL DERIV
                        self.e1_solve.append(-z_vals[1])
                        
                        intgr.integrate_step(tmp, self.h)
                        drv.deriv_step(tmp, self.h)

        #Функция находит точку, когда процесс становится устойчивым
        def find_dot_d(self, e):
                self.delta_e = 0.03
                self.test_e = e[-1]
                for i in range(len(e)):
                        if ((e[i] > self.test_e*self.delta_e - self.test_e) and (e[i] < self.test_e*self.delta_e + self.test_e)):
                                self.result = [i, e[i]]
                                return self.result

        #Новая интегральная оценка с желаемым Т
        def calculate_J(self):
                J=0
                #for i in range(len(self.z0_solve)-1):
                #        J += (((self.z0_solve[-1] - self.z0_solve[i])+self.tj**2*(self.z1_solve[-1]-self.z1_solve[i]))**2) * self.h 
                for i in range(len(self.e0_solve) - 1):
                        J += (self.e0_solve[i]**2 + self.Tr**2 * self.e1_solve[i]**2) * self.h 
                return J

        def plot_solution(self):
                X = np.linspace(0, self.x_limit, len(self.z0_solve))
                plt.plot(X, self.z0_solve)
                plt.xlabel("Время, с")
                plt.ylabel("Угол поворота оси, рад")
                plt.grid()
                plt.show()