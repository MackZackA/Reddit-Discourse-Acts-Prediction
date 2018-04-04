# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tokenizer for CountVectorizer/TfidfVectorizer

Uses nltk.tokenize.TweetTokenizer to keep punctuations and expression marks
in reddit posts
Not removing any certain pattern(e.g., url, numeric, ...) as sklearn
vectorizer can build its own vocabulary(dictionary) with a minimum document
frequency
"""

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer

stemmer = PorterStemmer()


def stem_tokens(tokens, porter_stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(porter_stemmer.stem(item))
    return stemmed


tweet_tokenizer = TweetTokenizer(strip_handles=True,
                                 preserve_case=False,
                                 reduce_len=True)  # e.g. waaaayyyyyy -> waayyy


def tokenizer(text):
    # remove punctuations other than ?!.
    # remove urls
    # text = re.sub(
    #     r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,'
    #     r'.;]+[-A-Za-z0-9+&@#/%=~_|][\r\n]*',
    #     ' ', text, flags=re.MULTILINE)
    tokens = tweet_tokenizer.tokenize(text)

    # stems = stem_tokens(tokens, stemmer)
    # return stems

    return tokens
