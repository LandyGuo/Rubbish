# coding=utf-8
import math
import codecs
import copy
import json
import sys

from collections import defaultdict


def count_ngram(lines, ngram):
    counter = {}
    for i, line in enumerate(lines):
        # add padding
        padding_prefix = '$[&'[:ngram-1]
        padding_end = padding_prefix[::-1].replace('[', ']')
        line = padding_prefix + line + padding_end
        for i in range(len(line) - ngram + 1):
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
        # if ln >= 2000: break
        lines.append(line)
    return lines


class NGramModel(object):

    def __init__(self, ngrams=[], model=None):
        assert ngrams, 'ngrams should be a list, eg [2], [2,3], [3,4] etc'
        self.ngrams = ngrams
        self.params = {}
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
        # 建立ngram词表
        ngram_freq, word_freq = {}, []
        for ngram in range(1, max(self.ngrams)+1):
            ngram_freq[ngram] = count_ngram(corpus, ngram)
            word_freq+=ngram_freq[ngram] # 合并ngram词表

        # 按照出现频次倒排
        word_freq.sort(key=lambda x: x[1], reverse=True)

        # 去掉出现频次5次以下的词
        word_freq = [x for x in word_freq if x[1]>=5]

        self.word_size = len(word_freq)

        # 建立词和id之间的映射
        self.id2word = dict([(id, word[0])
                             for id, word in zip(range(self.word_size), word_freq)])
        self.word2id = dict(map(lambda x: x[::-1], self.id2word.items()))

        # 计算bigram prob
        self.prob_matrix = defaultdict(dict)

        for ngram in self.ngrams:
            assert ngram >= 2, 'ngram should gt 2!, but got {} instead'.format(ngram)
            n_freq = dict(ngram_freq[ngram])
            n_1_freq = dict(ngram_freq[ngram-1])

            for k in n_freq.keys():
                if len(k)!=ngram: continue
                id0, id1 = self.word2id.get(
                    k[:ngram-1], self.word_size), self.word2id.get(
                    k[-1], self.word_size)
                # k0k1出现次数/k0出现次数
                k0_cnt = n_1_freq.get(k[:ngram-1], 0)
                k0k1_cnt = n_freq.get(k, 0)
                self.prob_matrix[str(id0)][str(id1)] = math.log((k0k1_cnt * 1.0 + 1.0) / (k0_cnt + len(n_1_freq)))

        self.save(word_size=self.word_size,
                  word2id=self.word2id,
                  prob_matrix=self.prob_matrix,
                  ngrams = self.ngrams)
        self.dump(save_model)


    def predict(self, sent):
        ori_sent = sent.strip()

        # padding sentence
        prob, normal = 0, 0
        # print("sent:%s"%sent.encode('utf-8'))
        for ngram in self.params['ngrams']:
            padding_prefix = '$[&'[:ngram - 1]
            padding_end = padding_prefix[::-1].replace('[', ']')
            sent = padding_prefix + ori_sent + padding_end
            

            # print("-------------------------ngrams:%s-----------------------"%ngram)
            # print("sentence:%s"%sent)
            p = 0
            for i in range(ngram-1, len(sent)):
                cond_p = self.condition_prob(sent[i-ngram+1:i], sent[i])
                # print("{} gram: {}->{}:{}".format(ngram, sent[i-ngram+1:i].encode('utf-8'), sent[i].encode('utf-8'), cond_p))
                p += cond_p
            prob += p
            normal += len(sent) - ngram+1
        return prob*1.0/normal


    def condition_prob(self, x, y):  # p(y/x)
        x_id, y_id = self.params['word2id'].get(
            x, self.params['word_size']), self.params['word2id'].get(
            y, self.params['word_size'])
        return self.params['prob_matrix'].get(str(x_id), {}).get(str(y_id), -20)



if __name__ == "__main__":
    corpus_path = sys.argv[1]
    corpus = load_txt(corpus_path)
    lm = NGramModel(ngrams=[2,3,4], model=None)
    lm.train(corpus, 'hospital.234gram.20180828')

    test_cases = [u'收费专用章', u'楼番禺区妇药团*药', u'医*第天', u'宁波市城镇职工基本医疗', u'综合医院外科门诊', u'门港医专疗疗诊费发票',u'沈阳皇姑国防医院',u'桂号票山天发**']
    for case in test_cases:
        score = lm.predict(case)
        pred = 1 if score < -10.0 else 0  # 1 有问题的医院名称 ； 0  正常医院名称
        print("case:%s score:%s result:%s" % (case, score, pred))
