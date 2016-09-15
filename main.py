#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
# import os
# import timeit

import numpy as np

# TODO:
# 1. Baixar o Dataset Bag of Words, arquivo NYTimes, do UCI
# 2. Crie bag of words para os primeiros 1.000 documentos, usando o vocabulário de 102.650 termos.
# 3. Calcular a distância entre cada par de vetores_de_bag_of_words acima (força bruta). Anote o
#    tempo gasto usando a biblioteca timeit, chamando a função timeit.time() antes e depois e
#    fazendo a subtração. Guarde os valores também.
# 4. Para n = 4, 16, 64, 256, 1024, 4096, 15768, 65536 faça:
#  a) Obtenha a matriz n x d pelo método de Achlioptas (use
#       options = (-1., 0., 1.)
#       prob_weights = (1./6., 2./3., 1./6.)
#       np.random.choice(options, size=(n, d), p=prob_weights))
#     e pelo método da gaussiana (use
#       numpy.random.normal(loc=0.0, scale=1.0/math.sqrt(n), size=(n, d))).
#     meça o tempo das gerações e guarde-o (ver item 3 para dica).
#  b) Projete os documentos em R^n (basta fazer
#       np.dot(random_projection_matrix, original_samples_matrix))
#     onde original_samples_matrix tem uma sample em cada COLUNA (ou seja, ela é d x N). Meça o
#     tempo e guarde-o (vide item 3)
#  c) Calcule as distâncias entre as projeções. Meça o tempo (vide item 3) e guarde-o.
#  d) Calcule a distorção máxima encontrada (use a função calculate_dataset_max_distortion nas
#     distâncias calculadas na letra (c) acima).
#  e) Calcule a distorção máxima previsto pelo lema de J-L (use calculate_max_distortion_probability
#     pra descobrir a probabilidade de uma das distorções ter sido tão grande quanto a encontrada)
# 5. Criar slides explicando a solução e resultados.
#
# obs: para guardar os tempos seria bom definirmos uma matriz de dimensão
#      (numero_de_valores_de_n x numero_de_medicoes_de_tempo_que_fazemos). Somamos 1 pq
#      medimos o tempo pro caso sem projeção. A dimensão no caso acima seria 8 x 4. Precisamos
#      também de mais uma variável pra guardar o tempo pro caso sem projeção.

# Notação usada no código (para tornar as variáveis mais explicativas):
# n = projected_sample_dimension
# d = original_sample_dimension,
# N = number_of_samples

def calculate_sample_distortion(original_vector, projected_vector):
    return abs(
        (np.dot(projected_vector, projected_vector) / np.dot(original_vector, original_vector)) - 1)

def calculate_max_distortion(list_of_original_vectors, list_of_projected_vectors):
    max_distortion = 0.0
    for original_vector, projected_vector in zip(
            list_of_original_vectors, list_of_projected_vectors):
        curr_distortion = calculate_sample_distortion(original_vector, projected_vector)
        if curr_distortion > max_distortion:
            max_distortion = curr_distortion
    return max_distortion

def calculate_max_distortion_probability(number_of_samples, original_sample_dimension,
                                         projected_sample_dimension, max_distortion):
    return float(number_of_samples**2) / math.exp(
        (projected_sample_dimension * max_distortion**2)/6.0)




if __name__ == '__main__':
    # TODO: definir o nome dos arquivos a serem abertos e chamar as funções na ordem necessária.
    # Depois apagar o "pass"
    pass
