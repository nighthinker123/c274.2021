# Copyright 2020-2021 Paul Lu
# Test 02.  Use functionality for Tasks I, II, and III.
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
    tset = TrainingSet()         # Using NEW functionality
    run1 = PRAClassifyByTopN()   # New class based on new class ClassifyByTopN()

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
    print(" Test 02.  Testing I, II, and III ")
    print("--------------------------------------------")

    pfeatures = tset.get_env_variable("pos-features")
    print("pos-features: ", pfeatures)
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)

    print("\n** Full Dataset (no preprocessing) *********")
    run1.target_top_n(tset, num=3, label=plabel)
    run1.classify_all(tset)
    run1.print_config()
    run1.eval_training_set(tset, plabel, lines=False)
    run1.print_pra()

    print("\n** Full Dataset (with preproc, top N) ******")
    tset.preprocess()
    run1.target_top_n(tset, num=3, label=plabel)
    run1.classify_all(tset)
    run1.print_config()
    run1.eval_training_set(tset, plabel, lines=False)
    run1.print_pra()

    print("\nDoes preprocessing improve PRA? (Not 100% apples vs. oranges.)")

    print("\n** By folds (3) ****************************")

    # Create folds for cross-validation.  Not necessarily most efficient way..
    ts_3 = tset.return_nfolds(3)

    # Done this (inefficient) way to clearly show cross validation
    #   The following really shows off the encapsulation/ADT benefits of OO
    train0 = TrainingSet()
    test0  = TrainingSet()
    test0.add_training_set(ts_3[0])
    train0.add_training_set(ts_3[1])
    train0.add_training_set(ts_3[2])

    train1 = TrainingSet()
    test1  = TrainingSet()
    train1.add_training_set(ts_3[0])
    test1.add_training_set(ts_3[1])
    train1.add_training_set(ts_3[2])

    train2 = TrainingSet()
    test2  = TrainingSet()
    train2.add_training_set(ts_3[0])
    train2.add_training_set(ts_3[1])
    test2.add_training_set(ts_3[2])

    print("   *** Using Test Fold 0 *******************")
    run1.target_top_n(train0, num=3, label=plabel)
    run1.classify_all(test0)
    run1.print_config()
    run1.eval_training_set(test0, plabel, lines=False)
    run1.print_pra()

    print("   *** Using Test Fold 1 *******************")
    run1.target_top_n(train1, num=3, label=plabel)
    run1.classify_all(test1)
    run1.print_config()
    run1.eval_training_set(test1, plabel, lines=False)
    run1.print_pra()

    print("   *** Using Test Fold 2 *******************")
    run1.target_top_n(train2, num=3, label=plabel)
    run1.classify_all(test2)
    run1.print_config()
    run1.eval_training_set(test2, plabel, lines=False)
    run1.print_pra()
    return


if __name__ == "__main__":
    main()
