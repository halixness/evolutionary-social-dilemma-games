#!/usr/bin/python3
"""
Implementation of a behaviour tree based on python code

Author: Leonardo Lucio Custode
Creation Date: 08-04-2020
Last modified: mar 5 mag 2020, 23:30:45
"""
from .decision_tree import DecisionTree


class PythonDT(DecisionTree):
    def __init__(self, phenotype, leaf):
        super(PythonDT, self).__init__()
        self.program = phenotype
        self.leaves = {}
        n_leaves = 0

        while "_leaf" in self.program:
            new_leaf = leaf()
            leaf_name = "leaf_{}".format(n_leaves)
            self.leaves[leaf_name] = new_leaf

            self.program = self.program.replace("_leaf", "'{}.get_action()'".format(leaf_name), 1)
            self.program = self.program.replace("_leaf", "{}".format(leaf_name), 1)

            n_leaves += 1
        self.exec_ = compile(self.program, "<string>", "exec", optimize=2)
    
    def get_action(self, input):
        if len(self.program) == 0:
            return None
        variables = {}  # {"out": None, "leaf": None}
        for idx, i in enumerate(input):
            variables["_in_{}".format(idx)] = i
        variables.update(self.leaves)

        exec(self.exec_, variables)

        current_leaf = self.leaves[variables["leaf"]]
        current_q_value = max(current_leaf.q)
        if self.last_leaf is not None:
            self.last_leaf.update(self.current_reward, current_q_value)
        self.last_leaf = current_leaf 
        
        return current_leaf.get_action()
    
    def __call__(self, x):
        return self.get_action(x)

    def __str__(self):
        return self.program
