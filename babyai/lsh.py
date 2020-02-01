import argparse
import numpy as np
import os
import pickle
import torch
from torch.nn import functional as F
from copy import copy
from itertools import combinations
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
from transformers import *

class LSH:
    def __init__(self, data):
        self.data = data
        self.model = None

    def __generate_random_vectors(self, num_vector, dim):
        return np.random.randn(dim, num_vector)

    def train(self, num_vector, seed=None):
        dim = self.data.shape[1]
        if seed is not None:
            np.random.seed(seed)

        random_vectors = self.__generate_random_vectors(num_vector, dim)
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        table = {}

        # Partition data points into bins
        bin_index_bits = (self.data.dot(random_vectors) >= 0)

        # Encode bin index bits into integers
        bin_indices = bin_index_bits.dot(powers_of_two)

        # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
        for data_index, bin_index in enumerate(bin_indices):
            if bin_index not in table:
                # If no list yet exists for this bin, assign the bin an empty list.
                table[bin_index] = []
            # Fetch the list of document ids associated with the bin and add the document id to the end.
            table[bin_index].append(data_index)

        self.model = {'bin_indices': bin_indices, 'table': table,
                      'random_vectors': random_vectors, 'num_vector': num_vector}
        return self

    def __search_nearby_bins(self, query_bin_bits, table, search_radius=2, initial_candidates=set()):
        num_vector = self.model['num_vector']
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        # Allow the user to provide an initial set of candidates.
        candidate_set = copy(initial_candidates)

        for different_bits in combinations(range(num_vector), search_radius):
            alternate_bits = copy(query_bin_bits)
            for i in different_bits:
                alternate_bits[i] = 1 if alternate_bits[i] == 0 else 0

            # Convert the new bit vector to an integer index
            nearby_bin = alternate_bits.dot(powers_of_two)

            # Fetch the list of documents belonging to the bin indexed by the new bit vector.
            # Then add those documents to candidate_set
            if nearby_bin in table:
                candidate_set.update(table[nearby_bin])

        return candidate_set

    def query(self, query_vec, k, max_search_radius, initial_candidates=set()):

        if not self.model:
            print('Model not yet build. Exiting!')
            exit(-1)

        data = self.data
        table = self.model['table']
        random_vectors = self.model['random_vectors']

        bin_index_bits = (query_vec.dot(random_vectors) >= 0).flatten()

        candidate_set = set()
        # Search nearby bins and collect candidates
        for search_radius in range(max_search_radius + 1):
            # print('cand set: {}'.format(candidate_set))
            cand_set = self.__search_nearby_bins(bin_index_bits, table,
                                                      search_radius, initial_candidates=initial_candidates)
            for c in cand_set: 
                candidate_set.add(c)

        # Sort candidates by their true distances from the query
        nearest_neighbors = DataFrame({'id': list(candidate_set)})
        # import ipdb; ipdb.set_trace()
        candidates = data[np.array(list(candidate_set)), :]
        nearest_neighbors['distance'] = pairwise_distances(candidates, query_vec.reshape(1, -1), metric='cosine').flatten()

        return nearest_neighbors.nsmallest(k, 'distance')


class LSHBox:
    def __init__(self, embedding_type, sentence_list):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.words = sorted({word for sentence in sentence_list for word in sentence.split()})
        self.embedding_type = embedding_type

        self.sent_to_embedding = self.load_sentences(embedding_type, sentence_list)
        
        self.embedding_to_sent = {}
        data = []
        for sent, embed in self.sent_to_embedding.items():
            self.embedding_to_sent[str(embed)] = sent 
            data.append(embed.squeeze(0)) 

        data = [x.numpy() for x in data]
        self.data = np.asarray(data)
        self.lsh_model = LSH(self.data)
        num_of_random_vectors = 10
        self.lsh_model.train(num_of_random_vectors)

    def load_sentences(self, embedding_type, sentences): 
        all_info = {}
        for sentence in sentences: 
            input_ids = torch.tensor([self.tokenizer.encode(sentence)])  

            out = []

            if embedding_type in {"bert", "both"}:
                with torch.no_grad():
                    _, _, hiddens = self.model(input_ids)
                    word_rep = hiddens[0].mean(dim=1)
                    seq_rep = hiddens[-1].mean(dim=1)
                out.append(F.normalize(word_rep, dim=1))
                out.append(F.normalize(seq_rep, dim=1))

            if embedding_type in {"lex", "both"}:
                lex = torch.tensor([[1 if word in sentence.split() else 0 for word in self.words]]).float()
                out.append(F.normalize(lex, dim=1))

            embedding = torch.cat(out, dim=1)
            all_info[sentence] = embedding 
        
        return all_info

    def get_single_embedding(self, sentence): 
        input_ids = torch.tensor([self.tokenizer.encode(sentence)])  
        out = []

        if self.embedding_type in {"bert", "both"}:
            with torch.no_grad():
                _, _, hiddens = self.model(input_ids)
                word_rep = hiddens[0].mean(dim=1)
                seq_rep = hiddens[-1].mean(dim=1)
            out.append(F.normalize(word_rep, dim=1))
            out.append(F.normalize(seq_rep, dim=1))

        if self.embedding_type in {"lex", "both"}:
            lex = torch.tensor([[1 if word in sentence.split() else 0 for word in self.words]]).float()
            out.append(F.normalize(lex, dim=1))

        embedding = torch.cat(out, dim=1)
        return embedding 

    def query(self, sentence): 
        similarity_meta_LSH = []
        embed = self.get_single_embedding(sentence)
        res = self.lsh_model.query(embed.squeeze(0).numpy(), 5, 10)
        y = res['id'].tolist() 
        keys = [str(torch.from_numpy(self.data[x]).unsqueeze(0)) for x in y]  
        results = [self.embedding_to_sent[key] for key in keys] 
        similarity_meta_LSH.append((sentence, results))
        closest = results[0]
        return results[0]


def main(args): 
    lsh_model = LSHBox(args.sentences, args.embeddings)
    test_sentence = 'dogs are cute'
    print(lsh_model.query(test_sentence))
       


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences', type=str, default='sentences.txt')
    parser.add_argument('--embeddings', type=str, default='BERT')
    args = parser.parse_args()
    main(args)
