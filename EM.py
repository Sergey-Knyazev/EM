#!/usr/bin/env python3
"""
Author: Sergey Knyazev
Email: sergey.n.knyazev@gmail.com
Created: 11/9/2019
"""

import numpy as np
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="Implementations of EM for EM_Pathways and CliqueSNV tools")
    parser.add_argument('-H', '--h_matrix_in', required=True, type=str, dest='h_in_csv',
                        help='a csv file with initial H matrix')
    parser.add_argument('-F', '--f_vector', required=True, type=str, dest='f_csv',
                        help='a csv file with F vector')
    parser.add_argument('-G', '--g_vector_in', required=True, type=str, dest='g_in_csv',
                        help='a csv file with initial G vector')
    parser.add_argument('-o', '--h_matrix_out', required=True, type=str, dest='h_out_csv',
                        help='an output csv file with H matrix')
    parser.add_argument('-g', '--g_vector_out', required=True, type=str, dest='g_out_csv',
                        help='an output csv file with G vector')
    parser.add_argument('-e', '--epsilon', type=float, dest='e', default=float(1e-6),
                        help='EM convergence parameter')
    parser.add_argument('-m', '--one_step_em', action='store_true', dest='m',
                        help='If set, then run one-step EM, otherwise two-step EM')
    return parser.parse_args()


def em1_step(H, G, F):
    E = ((H*G).T*F).T
    E = (E.T/E.sum(axis=1)).T
    return E.sum(axis=0)/E.sum()


def em2_step(H, G, F):
    F_exp = np.matmul(H, G)
    F_exp = F_exp/F_exp.sum()
    return (H.T*F/F_exp).T


def em_two_steps(H, G, F, e):
    #EM1
    while True:
        G_new = em1_step(H, G, F)
        if np.linalg.norm(G-G_new) < e:
            break
        G = G_new
    #EM2
    while True:
        H_new = em2_step(H, G, F)
        if np.linalg.norm(H-H_new) < e:
            break
        H = H_new
    return G_new, H_new


#1 step EM
def one_step_em(H, G, F, e):
    while True:
        G_new = em1_step(H, G, F)
        if np.linalg.norm(G-G_new) < e:
            return G_new
        G = G_new


#2 step em
def two_step_em(H, G, F, e):
    while True:
        G_new, H_new = em_two_steps(H, G, F, e)
        if np.linalg.norm(G-G_new) < e:
            return G_new, H_new
        G = G_new
        H = H_new


if __name__ == "__main__":
    args = arg_parse()
    H = np.loadtxt(args.h_in_csv, delimiter=',')
    G = np.loadtxt(args.g_in_csv, delimiter=',')
    F = np.loadtxt(args.f_csv, delimiter=',')

    H = H / H.sum(axis=0)
    G = G / G.sum()
    F = F / F.sum()

    if args.m:
        G_new = one_step_em(H, G, F, args.e)
        np.savetxt(args.g_out_csv, G_new, delimiter=',')
        np.savetxt(args.h_out_csv, H, delimiter=',')
    else:
        G_new, H_new = two_step_em(H, G, F, args.e)
        np.savetxt(args.g_out_csv, G_new, delimiter=',')
        np.savetxt(args.h_out_csv, H_new, delimiter=',')
