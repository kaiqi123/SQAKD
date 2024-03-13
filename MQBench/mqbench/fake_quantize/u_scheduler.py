import numpy as np
import math
import sys


class ConstantU_Scheduler:
    def __init__(self, u_init, u_max, updating_steps=None):
        self.u_init = u_init
        self.u_max = u_max
        self.updating_steps = updating_steps
        self.step_counter = 0

    def step(self):
        self.step_counter += 1
        if self.step_counter >= self.updating_steps:
            self.u = self.u_max
        else:
            self.u = self.u_init
        return self.u

class LinearU_Scheduler:
    def __init__(self, u_init, u_max, k):
        self.u_init = u_init
        self.u_max = u_max
        self.step_counter = 0
        self.k = k

    def step(self):
        self.step_counter += 1
        self.u = min(self.k * self.step_counter + self.u_init, self.u_max)
        return self.u

class LogU_Scheduler:
    def __init__(self, u_base, u_max, k, log_base):
        self.u_base = u_base
        self.u_max = u_max
        self.step_counter = 0
        self.k = k
        self.log_base = log_base

    def step(self):
        self.step_counter += 1
        self.u = min(math.log(self.k * self.step_counter + self.u_base, self.log_base), self.u_max)
        return self.u

class ExpU_Scheduler:
    def __init__(self, u_base, u_max, k, exp_base):
        self.u_base = u_base
        self.u_max = u_max
        self.step_counter = 0
        self.k = k
        self.exp_base = exp_base

    def step(self):
        self.step_counter += 1
        self.u = min(self.exp_base ** (self.k * self.step_counter + self.u_base), self.u_max)
        return self.u

class LinearAndExpU_Scheduler:
    def __init__(self, u_init, u_max, k, steps_for_linear, exp_base, exp_k, u_at_linear_step):
        self.u_init = u_init
        self.u_max = u_max
        self.step_counter = 0
        self.k = k
        self.steps_for_linear = steps_for_linear
        # for exp
        self.exp_base = exp_base
        self.exp_k = exp_k
        self.u_at_linear_step = u_at_linear_step

    def step(self):
        self.step_counter += 1
        if self.step_counter <= self.steps_for_linear:
            self.u = min(self.k * self.step_counter + self.u_init, self.u_max)
        else:
            step_count_exp = self.step_counter - self.steps_for_linear
            self.u = min(self.exp_base ** (self.exp_k * step_count_exp + self.u_at_linear_step), self.u_max)
        return self.u


class CosineU_Scheduler:
    def __init__(self, u_init, u_max, steps_for_updating):
        self.u_init = u_init
        self.u_max = u_max
        self.steps_for_updating = steps_for_updating
        self.step_counter = 0

    def step(self):
        self.step_counter += 1
        if self.step_counter <= self.steps_for_updating:
            self.u = self.u_max + (self.u_init - self.u_max) * (1 + math.cos((self.step_counter) * math.pi / self.steps_for_updating)) / 2
        else:
            self.u = self.u_max
        return self.u


class ExpU_Up_Down_Scheduler:
    def __init__(self, u_base, u_max, k, exp_base, steps_for_increasing, steps_for_updating):
        self.u_base = u_base
        self.u_max = u_max
        self.step_counter = 0
        self.k = k
        self.exp_base = exp_base
        self.steps_for_increasing = steps_for_increasing
        self.steps_for_updating = steps_for_updating

    def step(self):
        self.step_counter += 1
        if self.step_counter <= self.steps_for_increasing:
            self.u = min(self.exp_base ** (self.k * self.step_counter + self.u_base), self.u_max)
        else:
            self.u = self.exp_base ** (self.k * (-self.step_counter+self.steps_for_updating) + self.u_base)
        return self.u