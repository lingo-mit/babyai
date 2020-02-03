import os
import pickle
import argparse
import numpy as np
import os
import pickle
import torch
import spacy
import scipy 
import collections 
import nltk 

from torch.nn import functional as F
from copy import copy
from itertools import combinations
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
from transformers import *
from .lsh import *
from nltk.parse.generate import generate, demo_grammar
#from nltk import CFG
#from nltk import Tree
#from nltk.parse import CoreNLPParser
#from pycorenlp import StanfordCoreNLP
from numpy.random import seed, random
from scipy.linalg import norm
from .templates import * 


def get_subtrees(text): 
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    subtexts = []
    for chunk in doc.noun_chunks:
        subtexts.append(chunk.text)
    return subtexts


# TODO make it so that the score counting is weighted based on how close to the top of the list the items are 
def munkres_agreement(human, candidates, lshbox, lshbox_full): 
    subphrases = get_subtrees(human)
    subphrase_embed = [lshbox.get_single_embedding(x) for x in subphrases]
    human_embed = lshbox.get_single_embedding(human)
    cand_to_subphrase_embed = {}
    smallest = None
    min_cost = None 

    overall_dists = []
    for candidate in candidates: 
        cand_embed = lshbox.get_single_embedding(candidate)
        high_level_score = dist_cos(cand_embed, human_embed)
        overall_dists.append((candidate, high_level_score))

    overall_dists = sorted(overall_dists, key = lambda x: x[1])
        
    subphrase_dists = []
    for candidate in candidates: 
        cand_subphrases = list(get_subtrees(candidate))
        cand_subphrase_embed = [lshbox.get_single_embedding(x) for x in cand_subphrases]
        cand_to_subphrase_embed[candidate] = cand_subphrase_embed

        matrix = []

        for sub in cand_subphrase_embed: 
            dists = []
            for sub2 in subphrase_embed: 
                dist = dist_cos(sub, sub2).tolist()
                dists.append(dist)
            matrix.append(dists)


        cost = np.array(matrix)[0]
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        total = cost[row_ind, col_ind].sum()

        subphrase_dists.append((candidate, total))

    subphrase_dists = sorted(subphrase_dists, key = lambda x: x[1])

    scores = {}
    for i, (cand, dist) in enumerate(overall_dists): 
        if cand not in scores: 
            scores[cand] = 0
        scores[cand] += i

    for i, (cand, dist) in enumerate(subphrase_dists): 
        if cand not in scores: 
            scores[cand] = 0
        scores[cand] += i
    

    best = []
    best_score = None
    for cand, score in scores.items(): 
        if best_score is None or best_score == score: 
            best.append(cand)
            best_score = score
        if best_score is None or best_score > score: 
            best = [cand]
            best_score = score

    #closest_baseline = lshbox_full.query(human, 3)

    return best


def munkres(human, candidates, lshbox, lshbox_full): 
    subphrases = get_subtrees(human)
    subphrase_embed = [lshbox.get_single_embedding(x) for x in subphrases]
    human_embed = lshbox.get_single_embedding(human)
    cand_to_subphrase_embed = {}
    smallest = None
    min_cost = None 

    overall_smallest = None
    overall_smallest_dist = None 
    for candidate in candidates: 
        cand_embed = lshbox.get_single_embedding(candidate)
        high_level_score = dist_cos(cand_embed, human_embed)
        cand_subphrases = list(get_subtrees(candidate))
        cand_subphrase_embed = [lshbox.get_single_embedding(x) for x in cand_subphrases]
        cand_to_subphrase_embed[candidate] = cand_subphrase_embed

        matrix = []

        for sub in cand_subphrase_embed: 
            dists = []
            for sub2 in subphrase_embed: 
                dist = dist_cos(sub, sub2).tolist()
                dists.append(dist)
            matrix.append(dists)


        cost = np.array(matrix)[0]
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        total = cost[row_ind, col_ind].sum()

        final = high_level_score + total 
        if min_cost is None or final < min_cost: 
            min_cost = final
            smallest = candidate 

        #closest_baseline = lshbox_full.query(human, 3)

    return smallest 


def find_closest(human, candidates, lshbox, lshbox_full): 
    # get subphrases from human
    subphrases = get_subtrees(human)
    human_embed = lshbox.get_single_embedding(human)
    alpha = 0.3
    cand_to_score = dict()
    smallest = None
    smallest_cand = None
    for candidate in candidates: 
        cand_embed = lshbox.get_single_embedding(candidate)
        cand_subphrases = list(get_subtrees(candidate))

        high_level_score = dist_cos(cand_embed, human_embed)
        cum_subphrase_dist = 0
        if len(cand_subphrases) > len(subphrases):
            continue 
        
        for i, phrase in enumerate(subphrases): 
            x = lshbox.get_single_embedding(phrase)
            if len(cand_subphrases) > i: 
                cand_phrase = cand_subphrases[i]
                y = lshbox.get_single_embedding(cand_phrase)
                dist = dist_cos(x, y)
                cum_subphrase_dist += dist
        
        total = high_level_score + 0.1 * cum_subphrase_dist
       
        cand_to_score[candidate] = total
        if smallest is None or total < smallest: 
            smallest = total
            smallest_cand = candidate

    #closest_baseline = lshbox_full.query(human, 3)

    return smallest

def generate_candidates(full_template_, subtree_closest, human, lsh_box_full): 
    objects = []
    num_subs = len(subtree_closest)

    filler_subsets = {}
    subtree_summary = collections.Counter() 
    for sub, closest_points, subtree_type in subtree_closest: 
        subtree_summary[subtree_type] += 1
        for name, info in MAPPINGS.items(): 
            if subtree_type == name: 
                if name not in filler_subsets: 
                    filler_subsets[name] = []
                for point in closest_points: 
                    filler_subsets[name].append(point)

    for temp in full_template_: 
        temp = temp[0]
        full_template = DETAILED_TEMPLATES[INVERSE_TEMPLATES[temp]]
        words = full_template.split()
    
        # TODO go through template, find all holes, determine type. keep summary. 
        # if filled holes don't align with available subtrees, early exit. 

        all_candidates = ['']
        no_match = False 

        opt_indices = [i for i, x in enumerate(words) if "[?" in x]
        all_indices = [i for i, x in enumerate(words) if "[" in x]

        non_opt = np.setdiff1d(all_indices, opt_indices)
        fill_summary = len(non_opt)
        # fill_summary = collections.Counter()
        # for index in non_opt: 
        #     cleaned = words[index].replace('[', '').replace(']', '')
        #     if '|' in cleaned: 
        #         fill_summary[cleaned] += 1
        #     else: 
        #         for name in MAPPINGS.keys(): 
        #             if name in cleaned: 
        #                 fill_summary[name] += 1 
        

        # for item, count in fill_summary.most_common():
        #     subtree_count = 0 

        #     if '|' in item: 
        #         parts = item.split('|')
        #         for part in parts: 
        #             subtree_count = 
        #     subtree_summary[]

    
        for i, word in enumerate(words): 
            tmp = []
            if '[' in word: 
                cleaned = word.replace('[', '').replace(']', '')
                if '?' in cleaned: 
                    cleaned = cleaned.replace('?', '')

                    if cleaned in filler_subsets: 
                        optional_fillers = filler_subsets[cleaned]

                        tmp_cands = all_candidates[:]
                        for filler in optional_fillers: 
                            for cand in all_candidates: 
                                cand = cand + " " + filler
                                tmp.append(cand)
                        for cand in tmp_cands: 
                            tmp.append(cand)   
                    else: 
                        continue   

                elif '|' in cleaned: 
                    parts = cleaned.split('|')

                    all_fillers = []
                    inner_match = False 

                    for part in parts: 
                        if part in filler_subsets: 
                            for fill in filler_subsets[part]: 
                                all_fillers.append(fill)
                            inner_match = True 
                    
                    if not inner_match: 
                        no_match = True 
                        break 

                    for filler in all_fillers: 
                        for cand in all_candidates: 
                            cand = cand + " " + filler
                            tmp.append(cand)
                
                else: 
                    if cleaned not in filler_subsets: 
                        no_match = True 
                        break 

                    all_fillers = filler_subsets[cleaned]
                    for filler in all_fillers: 
                        for cand in all_candidates: 
                            cand = cand + " " + filler
                            tmp.append(cand)

            else: 
                for cand in all_candidates: 
                    cand = cand + " " + word
                    tmp.append(cand)
                
            all_candidates = tmp 

        
        if no_match:
            continue 

        if len(all_candidates) == 0: 
            return
    
        # print('subtree sum: {} filler sum: {}'.format(subtree_summary, fill_summary))
        total_count = 0
        for item, count in subtree_summary.items(): 
            if item != 'optional': 
                total_count += 1
            
        if fill_summary > total_count:
            # print('cgontinuing') 
            continue 
            
        tmp_box = LSHBox(all_candidates)
        
        final_closest = munkres_agreement(human, all_candidates, tmp_box, lsh_box_full)
        return final_closest

def project_onto_grammar(lshbox, lshbox_templates, dataset):
    for example_info in dataset:
        human = example_info["human"]
        subtrees = example_info["subtrees"]
        candidates = project_one(lshbox, lshbox_templates, human, subtrees)
        print('human: {} projection: {}\n\n'.format(human, candidates))

def project_one(lshbox, lshbox_templates, human, subtrees): 
    lsh_box_full = None

    # rank templates 
    temp_closest = lshbox_templates.query(human, 4)

    if len(subtrees) > 0: 
        subtree_closest = []
        
        for sub in subtrees: 
            closest = lshbox.query(sub, 5)
            closest_points = []
            subtree_type = None
            subtree_types = collections.Counter()

            for i, (point, dist) in enumerate(closest):
                point = " ".join(point.split())

                for name, info in MAPPINGS.items(): 
                    if point in info: 
                        subtree_types[name] += 1
                
                if i == 0 and 'NULL' in point: 
                    continue 

                if 'NULL' in point:
                    subtree_types['none'] += 1


            subtree_types = subtree_types.most_common(2)[0]
            subtree_type1 = subtree_types[0]
            subtree_type2 = subtree_types[1]

            if subtree_type1 == 'start' or subtree_type1 == 'none' or subtree_type2 == 'none':  
                continue 

            for point, dist in closest: 
                if point in MAPPINGS[subtree_type1]: 
                    closest_points.append(point)

            subtree_closest.append((sub, closest_points, subtree_type1))

        # generate likely translations 
        if len(subtree_closest) > 0: 
            candidates = generate_candidates(temp_closest, subtree_closest, human, lsh_box_full)
            return candidates


def main(args):
        labels = pickle.load(open('labels.pickle', 'rb'))
    
        sentences = []
        for example, label in labels: 
            if label == '3' or label == '4' or (label != '1' and label != '2'): 
                example_parts = example.split('\n')[1:4]
                sentences.append(example_parts[1])
            
        dataset = []
        overall = set()
        for sentence in sentences: 
            info_dict = {}
            info_dict['human'] = sentence
            all_trees = set()
            subtrees = get_subtrees(sentence)
            for tree in subtrees: 
                all_trees.add(str(tree))
                overall.add(str(tree))
            info_dict['subtrees'] = all_trees 
            dataset.append(info_dict) 

        f = open('sentences.txt', 'r')
        sentences = f.readlines() 
        
        if os.path.exists('lshbox.p'): 
            lsh_box = pickle.load(open('lshbox.p', 'rb'))
        else: 
            lsh_box = LSHBox(EVERYTHING + STARTERS, allow_none=True)
            pickle.dump(lsh_box, open('lshbox.p', 'wb'))
        
        if os.path.exists('lshboxtemp.p'): 
            lsh_box_templates = pickle.load(open('lshboxtemp.p', 'rb'))
        else: 
            lsh_box_templates =  LSHBox(TEMPLATES.values(), allow_none=False)
            pickle.dump(lsh_box_templates, open('lshboxtemp.p', 'wb'))

        if os.path.exists('lshboxfull.p'): 
            lsh_box_full = pickle.load(open('lshboxfull.p', 'rb'))
        else: 
            lsh_box_full = LSHBox(sentences)
            pickle.dump(lsh_box_full, open('lshboxfull.p', 'wb'))
                
        print('finding projection...')
        project_onto_grammar(lsh_box, lsh_box_templates, lsh_box_full, dataset)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences', type=str, default='sentences.txt')
    parser.add_argument('--embeddings', type=str, default='BERT')
    args = parser.parse_args()
    main(args)

