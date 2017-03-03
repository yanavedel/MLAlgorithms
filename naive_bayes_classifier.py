import numpy as np


class NaiveBayes:
    def __init__(self):
        self.word_list = {}
        self.category_list = {}
        self.stop_word = []


    def incr_word(self, word, category):
        if word in self.stop_word:
            return;


        if word in self.word_list:
            self.word_list[word]["total_word"] += 1

            try:
                self.word_list[word]["category"][category] += 1
            except:
                self.word_list[word]["category"][category] = 1
        else:
            self.word_list[word] = {"total_word": 1, "category": {category: 1}}

        if category in self.category_list:
            self.category_list[category] += 1
        else:
            self.category_list[category] = 1

    def train(self, category, words):
        for word in words.split():
            self.incr_word(word, category)

    def get_wordcount_incat(self, word, category):
        try:
            return self.word_list[word]["category"][category]
        except:
            return 0.0

    def get_wordcount(self, word):
        try:
            return self.word_list[word]["total_word"]
        except:
            return 0.0


    def classify(self, text):
        best_prop = None
        markas = "undefine"
        total_trainingword = float(sum(self.category_list.values()))

        for cl in self.category_list:
            categ_total = float(self.category_list.get(cl))
            categ_prop =  categ_total / float(total_trainingword)

            sumword_prop = 0.0
            for w in text.split():
                word_c = float(self.get_wordcount(w))
                word_incat = self.get_wordcount_incat(w, cl)

                if word_incat > 0:
                    prob = np.log((word_incat / word_c) / categ_prop)
                    sumword_prop += prob

            result = np.log(categ_prop) + sumword_prop
            if result > best_prop:
                best_prop = result
                markas = cl

        return markas