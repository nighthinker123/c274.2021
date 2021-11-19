# Copyright 2020-2021 Paul Lu
# Test 01.  Use functionality for Tasks I, II, and III.
import sys
import copy     # for deepcopy()

# This works if ooclassifier.py is complete and solves Tasks I, II, and III
from ooclassifier import *

# Debug = False   # Sometimes, print for debugging

class PRAClassifyByTopN(ClassifyByTopN):
    def __init__(self, lw=[]):
        self.type = str(self.__class__)
        super().__init__(lw)
        return

    def print_pra(self):
        tp, fp, tn, fn = self.get_TF()
        precision = float(tp) / float(tp + fp)
        recall = float(tp) / float(tp + fn)
        accuracy = float(tp + tn) / float(tp + tn + fp + fn)
        print("Accuracy:  %3.2g = " % accuracy, end='')
        print("(%d + %d) / (%d + %d + %d + %d)" % (tp, tn, tp, tn, fp, fn))
        print("Precision: %3.2g = " % precision, end='')
        print("%d / (%d + %d)" % (tp, tp, fp))
        print("Recall:    %3.2g = " % recall, end='')
        print("%d / (%d + %d)" % (tp, tp, fn))


def main():
    global Debug
    tset = TrainingSet()        # Using NEW functionality
    run1 = PRAClassifyByTopN()  # New class based on new class ClassifyByTopN()

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or default filename
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

    # Save a copy of original training set
    tsetOrig = tset.copy()

    print("--------------------------------------------")
    print(" Test 01.  Testing I, II, and III ")
    print("--------------------------------------------")
    pfeatures = tset.get_env_variable("pos-features")
    print("pos-features: ", pfeatures)
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)

    print("-- Original Classification -----------------")
    run1.set_target_words(pfeatures.strip().split())  # FIXME Could be cleaner
    #     FIXME We could make set_target_words() polymorphic
    run1.classify_all(tset)
    run1.print_config()
    run1.eval_training_set(tset, plabel, lines=False)
    run1.print_pra()

    print("-- Preprocessing and Top 3 Words -----------")
    tset.preprocess()
    run1.target_top_n(tset, num=3, label=plabel)  # Call to new Task II method

    run1.classify_all(tset)
    run1.print_config()
    run1.eval_training_set(tset, plabel, lines=False)
    run1.print_pra()

    print("-- By folds (2 folds) ----------------------------")
    ts_2 = tset.return_nfolds(2)

    print("**** Fold 0 ********************************")
    run1.classify_all(ts_2[0])
    run1.print_config()
    run1.eval_training_set(ts_2[0], plabel, lines=False)
    run1.print_pra()

    print("**** Fold 1 ********************************")
    run1.classify_all(ts_2[1])
    run1.print_config()
    run1.eval_training_set(ts_2[1], plabel, lines=False)
    run1.print_pra()

    return


if __name__ == "__main__":
    main()
