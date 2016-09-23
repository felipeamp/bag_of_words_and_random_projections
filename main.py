#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import os
import timeit

import numpy as np
from scipy import spatial as sp
from scipy import sparse


# 1. Baixar o Dataset Bag of Words, arquivo NYTimes, do UCI. Não colocar no repositório.
# 2. Crie bag of words para os primeiros 1.000 documentos, usando o vocabulário de 102.650 termos.
#    Colocar o arquivo pronto no repositório (a não ser que seja muito grande). O código para
#    separar os primeiros 1.000 documentos e o vocabulário deveria estar num arquivo diferente (e
#    não na main.py).
# 3. Calcular a distância entre cada par de vetores_de_bag_of_words acima (força bruta) e guardar em
#    original_distances. Anote o tempo gasto usando a biblioteca timeit, chamando a função
#    timeit.default_timer() antes e depois e fazendo a subtração. Guarde os valores também.
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
#  d) Calcule a distorção máxima encontrada (use a função calculate_max_distortion nas distâncias
#     calculadas na letra (c) acima).
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


def get_docs_bag_of_words(filepath, num_docs, num_words_in_vocabulary):
    # Each column is a document, each row is a word.
    docs_bag_of_words = np.zeros(shape=(num_words_in_vocabulary, num_docs), dtype=int)
    with open(filepath, "r") as file:
        for line in file:
            doc_id, word_id, count = line.split()
            try:
                doc_id = int(doc_id)
                word_id = int(word_id)
                count = int(count)
            except ValueError:
                print("ValueError: values of doc_id, word_id or count couldn't be converted to"
                      " integer.")
            docs_bag_of_words[word_id - 1, doc_id - 1] = count
    return docs_bag_of_words


def gen_achlioptas_matrix(projected_sample_dimension, original_sample_dimension):
    assert projected_sample_dimension > 0
    assert original_sample_dimension > 0
    assert projected_sample_dimension < original_sample_dimension

    scaling = math.sqrt(3.) / float(projected_sample_dimension)
    options = (-scaling, 0., scaling)
    prob_weights = (1./6., 2./3., 1./6.)
    ach_mat = np.random.choice(options,
                               size=(projected_sample_dimension, original_sample_dimension),
                               p=prob_weights)
    return ach_mat


def gen_gaussian_matrix(projected_sample_dimension, original_sample_dimension):
    assert projected_sample_dimension > 0
    assert original_sample_dimension > 0
    assert projected_sample_dimension < original_sample_dimension

    gauss_mat = np.random.normal(loc=0.0,
                                 scale=1.0 / math.sqrt(projected_sample_dimension),
                                 size=(projected_sample_dimension, original_sample_dimension))
    return gauss_mat


def project(projection_matrix, original_samples_matrix):
    assert projection_matrix.shape[1] == original_samples_matrix.shape[0]
    return np.dot(projection_matrix, original_samples_matrix)

def project_sparse_csr(projection_matrix_csr, original_samples_matrix):
    assert projection_matrix_csr.shape[1] == original_samples_matrix.shape[0]
    return sparse.csr_matrix.dot(projection_matrix_csr, original_samples_matrix)

def project_sparse_dok(projection_matrix_dok, original_samples_matrix):
    assert projection_matrix_dok.shape[1] == original_samples_matrix.shape[0]
    return sparse.dok_matrix.dot(projection_matrix_dok, original_samples_matrix)

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


def calculate_max_distortion_prob(number_of_samples, projected_sample_dimension, max_distortion):
    return float(number_of_samples**2) * math.exp(
        - (projected_sample_dimension * max_distortion**2)/6.0)


def main():
    N = 1000    # Number of samples
    d = 102660  # original_sample_dimension
    filepath_docs_bag_of_words = os.path.join("dataset", "docword.nytimes.txt_preprocessed.txt")
    orig_dist_time = 0.0
    gen_ach_time = 0.0
    gen_gauss_time = 0.0
    proj_ach_time = 0.0
    proj_gauss_time = 0.0
    proj_dist_ach_time = 0.0
    proj_dist_gauss_time = 0.0

    convert_ach_sparse_csr_time = 0.0
    convert_ach_sparse_dok_time = 0.0
    proj_ach_sparse_csr_time = 0.0
    proj_ach_sparse_dok_time = 0.0

    # Passo 2.
    docs_bag_of_words = get_docs_bag_of_words(filepath=filepath_docs_bag_of_words,
                                              num_docs=N,
                                              num_words_in_vocabulary=d)
    # Debug
    # for i, doc in enumerate(docs_bag_of_words.T):
    #     for j in range(d):
    #         if doc[j] != 0:
    #             print("%i %i %i" % (i + 1, j + 1, doc[j]))
    # exit()

    # Passo 3.
    time_initial = timeit.default_timer()
    # sp.distance.pdist wants the vectors in the matrix' rows, thus we transpose.
    original_distances = sp.distance.pdist(docs_bag_of_words.T, metric='euclidean')
    orig_dist_time = timeit.default_timer() - time_initial
    print("Time to calculate the original pairwise distances: ", orig_dist_time)

    # Passo 4.
    proj_dims = [4**x for x in range(1, 9)]
    for n in proj_dims:
        print("-----------------------------------")
        print("Projecting in", n, "dimensions")

        # 4.a.
        time_initial = timeit.default_timer()
        ach_mat = gen_achlioptas_matrix(n, d)
        gen_ach_time = timeit.default_timer() - time_initial

        time_initial = timeit.default_timer()
        gauss_mat = gen_gaussian_matrix(n, d)
        gen_gauss_time = timeit.default_timer() - time_initial

        print("Time to generate the projection matrices:")
        print("\tAchlioptas method:", gen_ach_time)
        print("\tGaussian method:", gen_gauss_time)

        # Sparse matrix try
        time_initial = timeit.default_timer()
        ach_mat_sparse_csr = sparse.csr_matrix(ach_mat)
        convert_ach_sparse_csr_time = timeit.default_timer() - time_initial

        time_initial = timeit.default_timer()
        ach_mat_sparse_dok = sparse.dok_matrix(ach_mat)
        convert_ach_sparse_dok_time = timeit.default_timer() - time_initial

        print("Time to convert the Achlioptas projection matrix to sparse representations:")
        print("\tCompressed Sparse Row representation:", convert_ach_sparse_csr_time)
        print("\tDictionary of Keys representation:", convert_ach_sparse_dok_time)

        # 4.b.
        time_initial = timeit.default_timer()
        proj_ach = project(ach_mat, docs_bag_of_words)
        proj_ach_time = timeit.default_timer() - time_initial

        time_initial = timeit.default_timer()
        proj_gauss = project(gauss_mat, docs_bag_of_words)
        proj_gauss_time = timeit.default_timer() - time_initial

        print("Projection times:")
        print("\tAchlioptas method:", proj_ach_time)
        print("\tGaussian method:", proj_gauss_time)

        # Sparse matrix try
        time_initial = timeit.default_timer()
        proj_ach = project_sparse_csr(ach_mat_sparse_csr, docs_bag_of_words)
        proj_ach_sparse_csr_time = timeit.default_timer() - time_initial

        time_initial = timeit.default_timer()
        proj_ach = project_sparse_dok(ach_mat_sparse_dok, docs_bag_of_words)
        proj_ach_sparse_dok_time = timeit.default_timer() - time_initial

        print("Projection times with sparse representations:")
        print("\tCompressed Sparse Row representation:", proj_ach_sparse_csr_time)
        print("\tDictionary of Keys representation:", proj_ach_sparse_dok_time)

        print("(Convertion + Projection) times for each sparse representation:")
        print("\tAchlioptas method:", convert_ach_sparse_csr_time + proj_ach_sparse_csr_time)
        print("\tGaussian method:", convert_ach_sparse_dok_time + proj_ach_sparse_dok_time)

        # 4.c.
        time_initial = timeit.default_timer()
        # sp.distance.pdist wants the vectors in the matrix' rows, thus we transpose.
        projected_distances_ach = sp.distance.pdist(proj_ach.T, metric='euclidean')
        proj_dist_ach_time = timeit.default_timer() - time_initial

        time_initial = timeit.default_timer()
        # sp.distance.pdist wants the vectors in the matrix' rows, thus we transpose.
        projected_distances_gauss = sp.distance.pdist(proj_gauss.T, metric='euclidean')
        proj_dist_gauss_time = timeit.default_timer() - time_initial

        print("Time to calculate the pairwise distances:")
        print("\tAchlioptas method:", proj_dist_ach_time)
        print("\tGaussian method:", proj_dist_gauss_time)

        #  4.d
        ach_max_distortion = calculate_max_distortion(original_distances,
                                                      projected_distances_ach)
        gauss_max_distortion = calculate_max_distortion(original_distances,
                                                        projected_distances_gauss)
        print("Greatest distortions:")
        print("\tAchlioptas method:", ach_max_distortion)
        print("\tGaussian method:", gauss_max_distortion)

        #  4.e
        prob_ach_distortion = calculate_max_distortion_prob(N, n, ach_max_distortion)
        prob_gauss_distortion = calculate_max_distortion_prob(N, n, gauss_max_distortion)
        print("Probability of such greatest distortions (or larger):"
              " (delta, according to J-L lemma)")
        print("\tAchlioptas method:", prob_ach_distortion)
        print("\tGaussian method:", prob_gauss_distortion)


if __name__ == '__main__':
    main()
