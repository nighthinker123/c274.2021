#--------------------------------------------
#   Name: Zixu(Andy) Yu
#   ID: 1722179
#   CCID: zixu3
#   CMPUT 274, Fall 2021
#
#   Assignment1: ooclassifier
#-------------------------------------------- 
# Copyright 2020-2021 Paul Lu
import sys
import copy  # for deepcopy()
import math

Debug = False  # Sometimes, print for debugging.  Overridable on command line.
InputFilename = "file.input.txt"
TargetWords = [
    'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
    'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
    'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
    '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
]


def open_file(filename=InputFilename):
    try:
        f = open(filename, "r")
        return (f)
    except FileNotFoundError:
        # FileNotFoundError is subclass of OSError
        if Debug:
            print("File Not Found")
        return (sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return (sys.stdin)


def safe_input(f=None, prompt=""):
    try:
        # Case:  Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # Case:  From file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF before strip()
                if Debug:
                    print("EOF")
                return ("", False)
        return (line.strip(), True)
    except EOFError:
        return ("", False)


class C274:
    def __init__(self):
        self.type = str(self.__class__)
        return

    def __str__(self):
        return (self.type)

    def __repr__(self):
        s = "<%d> %s" % (id(self), self.type)
        return (s)


class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        super().__init__()  # Call superclass
        # self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        return

    # FIXME:  Incomplete.  Finish get_TF() and other getters/setters.
    def get_TF(self):
        return (self.TP, self.FP, self.TN, self.FN)

    # TODO: Could use Use Python properties
    #     https://www.python-course.eu/python3_properties.php
    def set_target_words(self, lw):
        # Could also do self.targetWords = lw.copy().  Thanks, TA Jason Cannon
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return (self.targetWords)

    def get_allWords(self):
        return (self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return (self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return (self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self, printSorted=True):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords (%d): " % ln, end='')
        if printSorted:
            print(sorted(self.get_target_words()))
        else:
            print(self.get_target_words())
        return

    def print_run_info(self, printSorted=True):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        if printSorted:
            print(sorted(self.get_nonTarget()))
        else:
            print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag + "-------- Confusion Matrix --------")
        print(tag + "%10s | %13s" % ('Predict', 'Label'))
        print(tag + "-----------+----------------------")
        print(tag + "%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag + "%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag + "%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag + "%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag + "%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel, lines=True):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        # zip is good for parallel arrays and iteration
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            # Format nice output
            if lines:
                w = ' '.join(w.split())
            else:
                w = ' '.join(ti.get_words())
                w = lb + " " + w

            # TW = testing bag of words words (kinda arbitrary)
            print("TW %s: ( %10s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():  # FIXME Write predicate
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w  # FIXME Use first word, but change
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return (inClass, evidence)

    # Could use a decorator, but not now
    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return (cl, e)

    def classify_all(self, ts, update=True, tlabel="classify_all"):
        for ti in ts.get_instances():
            cl, e = self.classify(ti, update=update, tlabel=tlabel)
        return


class TrainingInstance(C274):
    def __init__(self):
        super().__init__()  # Call superclass
        # self.type = str(self.__class__)
        self.inst = dict()
        # FIXME:  Get rid of dict, and use attributes
        self.inst["label"] = "N/A"  # Class, given by oracle
        self.inst["words"] = []  # Bag of words
        self.inst["class"] = ""  # Class, by classifier
        self.inst["explain"] = ""  # Explanation for classification
        self.inst["experiments"] = dict()  # Previous classifier runs
        return

    def get_label(self):
        return (self.inst["label"])

    def get_words(self):
        return (self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):  # tlabel = tag label
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return ("N/A")
        else:
            return (cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return ("N/A")
        else:
            return (cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
            self, line, run=None,
            tlabel="read", inclLabel=False
    ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return (self)

    # copy from preprocess.py
    def lowerCase(self, x: str) -> str:
        '''Convert the string to lower case.

        Arguemnts:
            x (str): the string needed to be converted

        Returns:
            x.lower() (str): the converted string.
        '''
        return x.lower()

    def rmSymbols(self, x: str) -> str:
        '''keep all the chars that is alphanumeric

        Arguemnts:
            x (str): the string needed to be processed

        Returns:
            output (str): the processed string.
        '''
        output = ''.join(a for a in x if a.isalnum())
        return output

    def rmNumbers(self, x: str) -> str:
        '''remove the numbers in the string if the string is not all numbers

        Arguemnts:
            x (str): the string needed to be processed

        Returns:
            output (str): the processed string.
        '''

        # if the string is all digits
        if x.isdigit():
            return x
        else:
            # keep all the elements that is not a digit
            output = ''.join(a for a in x if not a.isdigit())
            return output

    def rmStopWord(self, x: str) -> str:
        '''remove the stop words 

        Arguemnts:
            x (str): the string needed to be checked

        Returns:
            x (str): return the word if it's not a stop word. 
                        Rreturn None if it is.
        '''
        stopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                     "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
                     "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
                     "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
                     "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                     "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
                     "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
                     "about", "against", "between", "into", "through", "during", "before", "after",
                     "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                     "under", "again", "further", "then", "once", "here", "there", "when", "where",
                     "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
                     "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                     "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

        # remove the word if it's a stopword
        if x in stopWords:
            return None
        else:
            return x

    def process(self, sent: list, mode: str) -> str:
        '''operate different process based on the mode

        Arguemnts:
            sent (str): the string needed to be processed
            mode (str): the mode of the process

        Returns:
            processedWords (list): the processed list of words
        '''
        processedWords = []

        # determine the mode of process
        if mode == "full":
            for ele in sent:
                a = self.lowerCase(ele)
                b = self.rmSymbols(a)
                c = self.rmNumbers(b)
                d = self.rmStopWord(c)
                processedWords.append(d)
        elif mode == "keep-digits":
            for ele in sent:
                a = self.lowerCase(ele)
                b = self.rmSymbols(a)
                d = self.rmStopWord(b)
                processedWords.append(d)
        elif mode == "keep-stops":
            for ele in sent:
                a = self.lowerCase(ele)
                b = self.rmSymbols(a)
                c = self.rmNumbers(b)
                processedWords.append(c)
        elif mode == "keep-symbols":
            for ele in sent:
                a = self.lowerCase(ele)
                c = self.rmNumbers(a)
                d = self.rmStopWord(c)
                processedWords.append(d)

        # remove the None element in the list and convert it to the string
        processedWords = list(filter(None, processedWords))

        return processedWords

    def preprocess_words(self, mode=''):
        '''call the method to preprocess the words and assign it 
        to the training instance

        Arguemnts:
            mode (str): the mode of the process
            
        Returns:
            None
        '''
        # make a list containing all the valid modes
        validModes = ['full', 'keep-digits', 'keep-stops', 'keep-symbols']

        # if there is no command line argument, then set the mode to deafualy
        if mode == '': mode = "full"

        # if the mode entered is invalid, print a error message
        if mode not in validModes:
            print("Error. Usage: python3 preprocess.py <mode>\
            \nThe modes are: 'keep-digits' or 'keep-stops' or 'keep-symbols'\
            \nIf no mode is entered, the program will run a full process")

            quit()

        # set the values to the object dictionary
        self.inst["words"] = self.process(self.inst["words"], mode)

        return


class TrainingSet(C274):
    def __init__(self):
        super().__init__()  # Call superclass
        # self.type = str(self.__class__)
        self.inObjList = []  # Unparsed lines, from training set
        self.inObjHash = []  # Parsed lines, in dictionary/hash
        self.variable = dict()  # NEW: Configuration/environment variables
        return

    def set_env_variable(self, k, v):
        self.variable[k] = v
        return

    def get_env_variable(self, k):
        if k in self.variable:
            return (self.variable[k])
        else:
            return ""

    def inspect_comment(self, line):
        if len(line) > 1 and line[1] != ' ':  # Might be variable
            v = line.split(maxsplit=1)
            self.set_env_variable(v[0][1:], v[1])
        return

    def get_instances(self):
        return (self.inObjHash)  # FIXME Should protect this more

    def get_lines(self):
        return (self.inObjList)  # FIXME Should protect this more

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")  # Not used
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            if len(line) == 0:  # Blank line.  Skip it.
                continue

            # Check for comments *and* environment variables
            if line[0] == '%':  # Comments must start with % and variables
                self.inspect_comment(line)
                continue

            # Save the training data input, by line
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    def preprocess(self, mode=''):
        '''preprocess a set of training instance
        iterate the words in all training instance in the traning set and 
        preprocess it based on different modes

        Arguemnts:
            mode (str): the mode of the process
            
        Returns:
            None
        '''
        
        for ti in self.get_instances():
            # process all the words in inObjHash
            ti.preprocess_words(mode)
        return

    def return_nfolds(self, num=3):
        '''Divides the training instances in the training set into different
        partitions or folds

        Arguemnts:
            num (int): the number of partition or fold
            
        Returns:
            nfolds (list): a list of TrainingSet objects
        '''
        nfolds = []
        start = 0
        for i in range(num):
            # new object
            fold = TrainingSet()
            # pointer on the current index
            ptr = start
            # while index not out of bound
            while ptr < len(self.inObjHash):
                # deep copy the training instance to the new TraningSet object
                fold.inObjList.append(copy.deepcopy(self.inObjList[ptr]))
                fold.inObjHash.append(copy.deepcopy(self.inObjHash[ptr]))
                ptr += num
            start += 1
            # contains multiple new TraningSet object
            nfolds.append(fold)
        return nfolds

    def copy(self):
        '''Make a deepcopy a TrainingSet object that contains the same
        attributes as the original object

        Arguemnts:
            None
            
        Returns:
            newTs (TrainingSet): returns a deepcopy of current object
        '''
        # make a new object
        newTs = TrainingSet()
        # make a deep copy of current object
        newTs = copy.deepcopy(self)
        return newTs

    def add_training_set(self, tset):
        '''Adds all the training instances of tset to the current
        TrainingSet object.

        Arguemnts:
            tset (TrainingSet): the object of TrainingSet that get copied
            
        Returns:
            None
        '''
        # iterate every training instance in the tset
        [self.inObjList.append(copy.deepcopy(x))for x in tset.get_lines()]
        [self.inObjHash.append(copy.deepcopy(x))for x in tset.get_instances()]

        return


class ClassifyByTopN(ClassifyByTarget):
    '''
    A class to replace the current list of target words with a new list of target words

    Arugements:
        This class is inheritated from ClassifyByTarget.

    Methods:
        1. target_top_n(self, tset, num=5, label='')
    '''
    def target_top_n(self, tset, num=5, label=''):
        '''replace the current list of target words with a new list of target words
        based on word frequency calculated of the words

        Arguemnts:
            tset (TrainingSet): traningSet object containing all the ti
            num (int): the number of target words(doesn't include ties)
            label (str): the target label

        Returns:
            None
        '''
        wordsBag = []
        newTargetWords = []
        for ti in tset.get_instances():
            # if the label of training instance is designated
            if ti.get_label() == label:
                if ti.get_words != '':
                    # store all the words in the current ti into the word bag
                    [wordsBag.append(x) for x in ti.get_words()]
                else:
                    continue
            else:
                continue

        # count all the words in the wordBag and store the words and
        # the number of appearance in a dictionary
        countDict = dict((x, wordsBag.count(x)) for x in set(wordsBag))
        # sort the dictionary based on the # of appearance
        sortedDict = dict(sorted(countDict.items(), key=lambda x: x[1], reverse=True))

        # parallel lists
        words = list(sortedDict.keys())
        counts = list(sortedDict.values())

        # cutOff value of the list
        cutOff = counts[num - 1]

        for ele in words:
            # if # of apperance of the words is greater than the cutOff
            if sortedDict[ele] >= cutOff:
                newTargetWords.append(ele)
            else:
                # the count list is sorted from large to small so we can stop
                break
        # set the value
        self.set_target_words(newTargetWords)


# Very basic test of functionality
def basemain():
    global Debug
    tset = TrainingSet()
    run1 = ClassifyByTarget(TargetWords)
    if Debug:
        print(run1)  # Just to show __str__
        lr = [run1]
        print(lr)  # Just to show __repr__

    argc = len(sys.argv)
    if argc == 1:  # Use stdin, or default filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            # Allow override of Debug from command line
            if f == "Debug":
                Debug = True
                continue
            if f == "NoDebug":
                Debug = False
                continue

            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    print("--------------------------------------------")
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)
    print("NOTE: Not using any target words from the file itself")
    print("--------------------------------------------")

    if Debug:
        tset.print_training_set()
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, plabel)

    return


if __name__ == "__main__":
    basemain()


