# coding= utf-8
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
from stop_words import get_stop_words
from multiprocessing import Process, Manager

stop_words = get_stop_words('russian')
#stemmer = SnowballStemmer('russian')
morph = pymorphy2.MorphAnalyzer()

class parallelizer(Process):

        def __init__(self, thread_id, thread_num, dict, filename, new_csv_binary = None):
            Process.__init__(self)
            self.thread_id = thread_id
            self.thread_num = thread_num
            self.dict = dict
            self.new_csv_binary = new_csv_binary
            self.filename = filename


        def func1(self):
            count_texts = 0
            csv = pd.read_csv(self.filename)
#             csv = csv.head(100)
            cyrillic_translit = {u'\u0410': 'A', u'\u0430': 'a',
                                 u'\u0411': 'B', u'\u0431': 'b',
                                 u'\u0412': 'V', u'\u0432': 'v',
                                 u'\u0413': 'G', u'\u0433': 'g',
                                 u'\u0414': 'D', u'\u0434': 'd',
                                 u'\u0415': 'E', u'\u0435': 'e',
                                 u'\u0401': 'Yo', u'\u0451': 'yo',
                                 u'\u0416': 'Zh', u'\u0436': 'zh',
                                 u'\u0417': 'Z', u'\u0437': 'z',
                                 u'\u0418': 'I', u'\u0438': 'i',
                                 u'\u0419': 'I', u'\u0439': 'i',
                                 u'\u041a': 'K', u'\u043a': 'k',
                                 u'\u041b': 'L', u'\u043b': 'l',
                                 u'\u041c': 'M', u'\u043c': 'm',
                                 u'\u041d': 'N', u'\u043d': 'n',
                                 u'\u041e': 'O', u'\u043e': 'o',
                                 u'\u041f': 'P', u'\u043f': 'p',
                                 u'\u0420': 'R', u'\u0440': 'r',
                                 u'\u0421': 'S', u'\u0441': 's',
                                 u'\u0422': 'T', u'\u0442': 't',
                                 u'\u0423': 'U', u'\u0443': 'u',
                                 u'\u0424': 'F', u'\u0444': 'f',
                                 u'\u0425': 'Kh', u'\u0445': 'kh',
                                 u'\u0426': 'Ts', u'\u0446': 'ts',
                                 u'\u0427': 'Ch', u'\u0447': 'ch',
                                 u'\u0428': 'Sh', u'\u0448': 'sh',
                                 u'\u0429': 'Shch', u'\u0449': 'shch',
                                 u'\u042a': 'b', u'\u044a': 'b',
                                 u'\u042b': 'Y', u'\u044b': 'y',
                                 u'\u042c': "b", u'\u044c': "b",
                                 u'\u042d': 'E', u'\u044d': 'e',
                                 u'\u042e': 'Iu', u'\u044e': 'iu',
                                 u'\u042f': 'Ia', u'\u044f': 'ia'}

            def transliterate(word, translit_table):
                converted_word = ''
                for char in word:
                    transchar = ''
                    if char in translit_table:
                        transchar = translit_table[char]
                    else:
                        transchar = char
                    converted_word += transchar
                return converted_word

            for index, row in csv.iterrows():
                if index % self.thread_num != self.thread_id:
                    continue
                temp_text = ""  # пустышка для склеивания текста
                s = str(row['txt']).split(' ')
                count_texts = count_texts + 1

                for word in s:
                    if word not in stop_words:
                        if word not in ('Клиент:', 'Оператор:', ' ', ')', '(', ':') and re.search('.\d{2}.\d{2}.',
                                                                                                  word) == None and word.isdigit() == False:
                            temp_text = temp_text + transliterate(morph.parse(word)[0].normal_form, cyrillic_translit) + ' '

                self.dict.update({index: temp_text.lower()})

        def run(self):
            self.func1()

def preprocess(filename, output='output.csv', delimeter=',', threads = 6):
    csv = pd.read_csv(filename, delimeter)
    manager = Manager()
    dict = manager.dict()
    processes = []

    for i in range(6):
        bot = parallelizer(i, threads, dict, filename)
        processes.append(bot)
        bot.run()
        bot.start()

    for p in processes:
        p.join()

    list_of_texts = []
    for val in dict.values():
        list_of_texts.append(val.encode('utf-8'))

    pd.Series(list_of_texts).to_csv(output, delimeter)
    return ("File " + output + " has been saved.")
