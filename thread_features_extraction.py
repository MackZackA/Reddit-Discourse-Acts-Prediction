#!/usr/bin/env python3

import sys
import copy
import json
import pickle
import nltk

data_file = sys.argv[1]
thread_number_file = sys.argv[2]
sum_thread_post_depth = 0
thread_dictionary = {}
data_dictionary = {}
initial_authors = {}
thread_branch_number = {}
thread_id_dict = {}
thread_comments = {}

def preprocess_data(data, threads, sum_thread_post_depth):
    '''
    This function puts all lines of raw data into dictionaries.

    Args:
        data (dict): dictionary that takes the first system argument.
        threads (dict): dictionary that takes the second system argument.
    '''
    with open(data) as read_data:
        for line in read_data:
            # data_list.append(line)
            read_line = json.loads(line)
            comment_id = read_line['id']
            if comment_id not in data_dictionary:
                data_dictionary[comment_id] = line
            sum_thread_post_depth += read_line['depth']

    with open(threads) as thread_data:
        for line in thread_data:
            read_line = json.loads(line)
            post_id = read_line['id']
            if post_id not in thread_dictionary:
                thread_dictionary[post_id] = line


def get_root(id, data_dict):
    """
    This function finds the thread of current post.
    It uses recursion to access the id of its upper-layer post,
    attaining the thread id, the id of the top layer.
    
    Args:
        id (str): id of the current post.
        data_dict (dict): processed data file.

    Returns:
        id (str): thread id.
    """
    if not id or id not in data_dict:
        return id
    line = json.loads(data_dict[id])
    parent_id = line['parent_id'].split('_')[1]
    # current_author = line['author']
    if (parent_id) and (parent_id in data_dict):
        get_root(parent_id, data_dict)
    else:
        return id
        # return id, current_author 

def load_data(data_dict, thrd_dict, thread_comments):
    """
    This funciton loads the processed data to fills the helper dictionaries.

    Args:
        data_dict (dict): data_dictionary.
        thrd_dict (dict): thread_dictionary.
    """
    for c_id in data_dict.keys():
        read_line = json.loads(data_dict[c_id])
        # thread_id, init_author = get_root(read_line['id'], data_dict) # helper function
        thread_id = get_root(read_line['id'], data_dict)
        init_author = 'DNE'
        initial_authors[c_id] = init_author
        thread_id_dict[read_line['id']] = thread_id
        if thread_id not in thread_comments:
            thread_comments[thread_id] = 1
        elif thread_id in thread_comments and read_line['depth'] == 1:
            thread_comments[thread_id] += 1 # collect number of branches 
    for p_id in thrd_dict:
        read_line = json.loads(thrd_dict[p_id])
        if read_line['id'] not in thrd_dict:
            thrd_dict[read_line['id']] = read_line['num_comments']


def add_features(data):
    """
    This function adds four features to each json object.
    The four features are 'thread_id', 'thread_is_self_post',
    'thread_branch_number', and 'thread_branch_length'.

    Args:
        data (dict): data dictionary.
    """
    """Add num_char, num_word, num_sent, parent_num_word, parent_num_sent, parent_num_char, parent_post_depth, parent_body"""

    # result = copy.deepcopy(data)
    # for id in result.keys():
    #     read_line = json.loads(result[id])
    result = {}
    for id in data.keys():
        read_line = json.loads(data[id])
        body = read_line['body']
        num_char = len(body)
        num_word = len(nltk.word_tokenize(body))
        num_sent = len(nltk.sent_tokenize(body))
        if 'num_char' not in read_line:
            read_line['num_char'] = num_char
        if 'num_word' not in read_line:
            read_line['num_word'] = num_word
        if 'num_sent' not in read_line:
            read_line['num_sent'] = num_sent
        depth_normalized = float(read_line['depth']) / (sum_thread_post_depth + 1)
        if 'depth_normalized' not in read_line:
            read_line['depth_normalized'] = depth_normalized
        thread_id = thread_id_dict[read_line['id']]
        read_line['thread_id'] = thread_id
        is_first_post = False
        is_init_author = False
        is_parent_author = False
        if 'is_first_post' not in read_line:
            if read_line['depth'] == 1:
                is_first_post = True
            read_line['is_first_post'] = is_first_post
        if 'is_initial_author' not in read_line:
            if read_line['author'] == initial_authors[id]:
                is_init_author = True
            read_line['is_initial_author'] = is_init_author 
        parent_id = read_line['parent_id']
        parent_author = None
        parent_body = 'DNE'
        parent_text = 'DNE'
        parent_num_char = 0
        parent_num_word = 0
        parent_num_sent = 0
        parent_post_depth = 0
        parent_post_depth_normalized = 0
        # if parent_id != thread_id and parent_id in result:
        if parent_id.split('_')[1] != thread_id and parent_id.split('_')[1] in data:
            parent_line = json.loads(data[parent_id.split('_')[1]])
            parent_author = parent_line['author']
            parent_body = parent_line['body']
            parent_text = parent_line['article']
            parent_num_char = len(parent_body)
            parent_num_word = len(nltk.word_tokenize(parent_body))
            parent_num_sent = len(nltk.sent_tokenize(parent_body))
            parent_post_depth = parent_line['depth']
            parent_post_depth_normalized = float(parent_post_depth) / (sum_thread_post_depth + 1) 
        if 'parent_text' not in read_line:
            read_line['parent_text'] = parent_text
        if 'is_parent_author' not in read_line:
            if read_line['author'] == parent_author:
                is_parent_author = True
            read_line['is_parent_author'] = is_parent_author
        if 'parent_num_char' not in read_line:
            read_line['parent_num_char'] = parent_num_char
        if 'parent_num_word' not in read_line:
            read_line['parent_num_word'] = parent_num_word
        if 'parent_num_sent' not in read_line:
            read_line['parent_num_sent'] = parent_num_sent
        if 'parent_post_depth' not in read_line:
            read_line['parent_post_depth'] = parent_post_depth
        if 'parent_post_depth_normalized' not in read_line:
            read_line['parent_post_depth_normalized'] = parent_post_depth_normalized
        read_line['thread_id'] = thread_id
        if 'thread_is_self_post' not in read_line:
            read_line['thread_is_self_post'] = False
        read_line['thread_branch_num'] = thread_comments[thread_id]
        if 'thread_comment_num' not in read_line:
            read_line['thread_comment_num'] = thread_comments[thread_id]
        read_line['thread_branch_len'] = (thread_comments[thread_id] + 1) * 1.0 / read_line['thread_branch_num']
        ### renaming features, old names removed in remove_features.py
        read_line['in_reply_to'] = read_line['parent_id']
        read_line['post_depth'] = read_line['depth']
        read_line['post_depth_normalized'] = read_line['depth_normalized']
        read_line['url'] = read_line['title_url']
        read_line['text'] = read_line['article']
        read_line['majority_type'] = 'agreement'
        ###
        ### removing repetitive names
        read_line.pop('parent_body', None) 
        read_line.pop('subreddit_id', None)
        read_line.pop('subreddit_id', None)
        read_line.pop('depth_normalized', None)
        read_line.pop('depth', None)
        read_line.pop('id', None)
        read_line.pop('ups', None)
        read_line.pop('article', None)
        read_line.pop('history', None)
        read_line.pop('created_utc', None)
        read_line.pop('id', None)
        read_line.pop('title_domain', None)
        read_line.pop('parent_id')
        read_line.pop('title_url', None)
        ###
        if id not in result:
            result[id] = read_line
    with open('patrick_dict.json', 'w') as out:
        json.dump(result, out)
    for i in result.keys():
        print("The number of keys:", len(result[i].keys()))
        print("The keys:", result[i].keys())
        break 


if __name__ == "__main__": 
    preprocess_data(data_file, thread_number_file, sum_thread_post_depth)
    load_data(data_dictionary, thread_dictionary, thread_comments)
    add_features(data_dictionary)   
 
