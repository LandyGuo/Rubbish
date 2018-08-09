# coding=utf-8
import math
import codecs
import copy
import json

from collections import defaultdict


def count_ngram(lines, ngram):
    counter = {}
    for i, line in enumerate(lines):
        # ngrams
        for i in range(len(lines) - ngram + 1):
            cur_n_gram = line[i:i + ngram]
            if cur_n_gram not in counter:
                counter[cur_n_gram] = 0
            counter[cur_n_gram] += 1
    lst = counter.items()
    return lst


def load_txt(corpus_file):
    lines = []
    for ln, line in enumerate(codecs.open(corpus_file, 'r', 'utf-8')):
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    return lines


class NGramModel(object):

    def __init__(self, ngrams=2, model=None):
        self.ngrams = ngrams
        self.params = {}
        self.debug = False
        if model:
            self.load(model)

    def _save(self, name, value):
        self.params[name] = copy.deepcopy(value)

    def save(self, **kwargs):
        for k, v in kwargs.items():
            self._save(k, v)

    def dump(self, filename):
        with open(filename, 'wb') as f:
            json.dump(self.params, f)

    def load(self, model_file):
        self.params = json.load(open(model_file, 'rb'))

    def train(self, corpus, save_model):
        word_prob = count_ngram(corpus, 1)
        ngram_prob = count_ngram(corpus, self.ngrams)
        word_prob.sort(key=lambda x: x[1], reverse=True)
        word_cnt = dict(word_prob)
        self.word_size = len(word_prob)
        self.id2word = dict([(id, word[0])
                             for id, word in zip(range(self.word_size), word_prob)])
        self.word2id = dict(map(lambda x: x[::-1], self.id2word.items()))
        self.word_prob_map = dict(
            [(k, math.log((1.0 * v + 1.0) / (self.word_size + 1.0))) for k, v in word_prob])
        self.prob_matrix = defaultdict(dict)

        for k, v in ngram_prob:
            if len(k) != 2: continue
            k0, k1 = self.word2id.get(
                k[0], self.word_size), self.word2id.get(
                k[1], self.word_size)
            self.prob_matrix[str(k0)][str(k1)] = math.log( (v * 1.0 + 1.0) / (word_cnt.get(self.id2word.get(k0, '#'),0) + self.word_size))

        self.save(word_size=self.word_size,
                  word2id=self.word2id,
                  prob_matrix=self.prob_matrix,
                  word_prob_map=self.word_prob_map)
        self.dump(save_model)

    def predict(self, sent):
        sent = sent.strip()
        p = self.params['word_prob_map'].get(
            sent[0], math.log(1.0 / self.params['word_size'] + 1))
        if len(sent) <= 1:
            return p
        for i in range(1, len(sent)):
            cond_p = self.condition_prob(sent[i - 1], sent[i])
            p += cond_p
        avg_p = p / (len(sent) - 1)
        return avg_p

    def condition_prob(self, x, y):  # p(y/x)
        x_id, y_id = self.params['word2id'].get(
            x, self.params['word_size']), self.params['word2id'].get(
            y, self.params['word_size'])
        return self.params['prob_matrix'].get(str(x_id), {}).get(str(y_id), -20)


if __name__ == "__main__":
    lm = NGramModel(model='hospital.2grams.model')

    test_cases = [u'test1', u'test2', u'test3']
    for case in test_cases:
        score = lm.predict(case)
        pred = 1 if score < -10.0 else 0  
        print("case:%s score:%s result:%s" % (case, score, pred))
