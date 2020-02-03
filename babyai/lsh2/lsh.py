import os
import pickle
import argparse
import numpy as np
import os
import pickle
import torch
import spacy
from torch.nn import functional as F
from copy import copy
from itertools import combinations
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
from transformers import *

#import nltk 
#from nltk.parse.generate import generate, demo_grammar
#from nltk import CFG
#from nltk import Tree
#from nltk.parse import CoreNLPParser
#from pycorenlp import StanfordCoreNLP

BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_MODEL = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
BERT_MODEL.eval()


def dist_cos(query, target):
    query = query / torch.norm(query, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    return (1 - (query * target).sum(dim=1)) / 2


class COSBox: 
    def __init__(self, sentences): 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BERT_MODEL
        
        self.sent_to_embedding = self.load_sentences(sentences) 
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
    

    def load_sentences(self, sentences): 
        all_info = {}
        for sentence in sentences: 
            try: 
                sentence = sentence.decode('UTF-8')
            except: 
                pass 
            # print('sentence: {}'.format(sentence))
            input_ids = torch.tensor([self.tokenizer.encode(sentence)])  
            out = []
            with torch.no_grad():
                _, _, hiddens = self.model(input_ids)
                word_rep = hiddens[0].mean(dim=1)
                seq_rep = hiddens[-1].mean(dim=1)
            out.append(F.normalize(word_rep, dim=1))
            out.append(F.normalize(seq_rep, dim=1))
            embedding = torch.cat(out, dim=1)
            all_info[sentence] = embedding 

        with open('embeddings.pkl', 'wb') as f: 
            pickle.dump(all_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return all_info
    
    def get_single_embedding(self, sentence): 
        input_ids = torch.tensor([self.tokenizer.encode(sentence)])  
        out = []
        with torch.no_grad():
            _, _, hiddens = self.model(input_ids)
            word_rep = hiddens[0].mean(dim=1)
            seq_rep = hiddens[-1].mean(dim=1)
        out.append(F.normalize(word_rep, dim=1))
        out.append(F.normalize(seq_rep, dim=1))
        embedding = torch.cat(out, dim=1)

        return embedding 

    def query(self, sentence, num): 
        input_embedding = self.get_single_embedding(sentence)
    
        closest = []
        num_to_return = num

        def argmax(pairs):
            return max(pairs, key=lambda x: x[0])

        furthest_within_closest = None
        for line, embedding in self.sent_to_embedding.items():
            # dist1 = numpy.linalg.norm(input_embedding - embedding)
            dist = dist_cos(input_embedding, embedding)

            if len(closest) < num_to_return:
                closest.append((dist, line))

            elif dist < furthest_within_closest[0]:
                closest.remove(furthest_within_closest)
                closest.append((dist, line))

            furthest_within_closest = argmax(closest)
        return sorted(closest, key=lambda x: x[0])
        # print("natural language input: {}, closest synthetic statements: {}".format(text, sorted(closest, key=lambda x: x[0])))


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
    def __init__(self, sentences_path, allow_none=False, embedding_type='BERT'):
        if embedding_type == 'BERT': 
            self.model = BERT_MODEL
            self.tokenizer = BERT_TOKENIZER
            
            self.sent_to_embedding = self.load_sentences(sentences_path, allow_none=allow_none) 
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
           
        else: 
            raise NotImplementedError

    def load_sentences(self, sentences, allow_none): 
        all_info = {}
        avg = []
        for sentence in sentences: 
            try: 
                sentence = sentence.decode('UTF-8')
            except: 
                pass 
            # print('sentence: {}'.format(sentence))
            input_ids = torch.tensor([self.tokenizer.encode(sentence)])  
            out = []
            with torch.no_grad():
                _, _, hiddens = self.model(input_ids)
                word_rep = hiddens[0].mean(dim=1)
                seq_rep = hiddens[-1].mean(dim=1)
            out.append(F.normalize(word_rep, dim=1))
            out.append(F.normalize(seq_rep, dim=1))
            embedding = torch.cat(out, dim=1)
            all_info[sentence] = embedding 
            avg.append(embedding)

        if allow_none: 
            avg_embed = sum(avg)/ len(avg)
        # import ipdb; ipdb.set_trace()
        # print('AVG EMBED: {}'.format(avg_embed))
            all_info['NULL'] = avg_embed

        with open('embeddings.pkl', 'wb') as f: 
            pickle.dump(all_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return all_info

    def get_single_embedding(self, sentence): 
        input_ids = torch.tensor([self.tokenizer.encode(sentence)])  
        out = []
        with torch.no_grad():
            _, _, hiddens = self.model(input_ids)
            word_rep = hiddens[0].mean(dim=1)
            seq_rep = hiddens[-1].mean(dim=1)
        out.append(F.normalize(word_rep, dim=1))
        out.append(F.normalize(seq_rep, dim=1))
        embedding = torch.cat(out, dim=1)

        return embedding 

    def query(self, sentence, num): 
        embed = self.get_single_embedding(sentence)
        res = self.lsh_model.query(embed.squeeze(0).numpy(), 5, 17)
        y = res['id'].tolist()  
        dists = res['distance'].tolist()
        keys = [str(torch.from_numpy(self.data[x]).unsqueeze(0)) for x in y]  
        results = [self.embedding_to_sent[key] for key in keys] 
        full_results = []
        for i, res in enumerate(results): 
            full_results.append((res, dists[i]))
        closest = full_results[0:num]
        return full_results[0:num]
