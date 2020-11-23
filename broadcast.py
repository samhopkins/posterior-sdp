import numpy as np
from scipy.stats import bernoulli
import random


def get_tree(depth,degree):
    tree = {}
    tree['children'] = []
    if depth > 0:
        for i in range(degree):
            tree['children'].append(get_tree(depth-1,degree))
    return tree

#assumes root is assigned, propagates to children
def broadcast(root,eps):
    for child in root['children']:
        coin = bernoulli.rvs(0.5+eps)
        if coin==1:
            child['sign'] = root['sign']
        else:
            child['sign'] = 1-root['sign']
        broadcast(child,eps)


def infer_root_p1(root,eps):
    if root['children'] == []:
        root['p1'] = root['sign']
    else:
        for child in root['children']:
            infer_root_p1(child,eps)

        weights1 = [(0.5+eps)*child['p1'] + (0.5-eps)*(1 - child['p1']) for child in root['children']]
        weights0 = [(0.5-eps)*child['p1'] + (0.5+eps)*(1-child['p1']) for child in root['children']]

        w1 = np.prod(weights1)
        w0 = np.prod(weights0)
        z = w1 + w0
        root['p1'] = w1/z

def assign_leaf_potentials(root,f):
    if root['children'] == []:
        if root['sign'] == 1:
            root['sign'] = f()
        else:
            root['sign'] = 1-f()
    else:
        for child in root['children']:
            assign_leaf_potentials(child,f)

def get_empirical_dist(depth,degree,eps,num_samples):
    emp_dist = []
    for i in range(num_samples):
        t = get_tree(depth,degree)
        t['sign'] = 1
        broadcast(t,eps)
        infer_root_p1(t,eps)
        emp_dist.append(t['p1'])
    return emp_dist
