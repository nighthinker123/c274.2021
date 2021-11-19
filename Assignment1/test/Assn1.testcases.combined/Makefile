#### Assumes that ooclassifier.py is in the current working directory

correctness:
	make unit
	make alltasks

all:
	make base
	make unit
	make alltasks

unit:
	make unit1
	make unit2
	make unit3

alltasks:
	make alltasks1
	make alltasks2

alltasks2:
	python3 a1.alltasks.v02.py NoDebug helpdata.v1 > helpdata.v1.alltasks.v02.out
	-diff -w helpdata.v1.alltasks.v02.out Output.expected/helpdata.v1.alltasks.v02.out.expected
	python3 a1.alltasks.v02.py NoDebug weatherdata.v1 > weatherdata.v1.alltasks.v02.out
	-diff -w weatherdata.v1.alltasks.v02.out Output.expected/weatherdata.v1.alltasks.v02.out.expected


alltasks1:
	python3 a1.alltasks.v01.py NoDebug helpdata.v1 > helpdata.v1.alltasks.v01.out
	-diff -w helpdata.v1.alltasks.v01.out Output.expected/helpdata.v1.alltasks.v01.out.expected
	python3 a1.alltasks.v01.py NoDebug weatherdata.v1 > weatherdata.v1.alltasks.v01.out
	-diff -w weatherdata.v1.alltasks.v01.out Output.expected/weatherdata.v1.alltasks.v01.out.expected

# Tests for Task I
unit1:
	python3 a1.task.1.unittest.v01.py NoDebug helpdata.v1 > helpdata.v1.task.1.out
	-diff -w helpdata.v1.task.1.out Output.expected/helpdata.v1.task.1.out.expected
	python3 a1.task.1.unittest.v01.py NoDebug weatherdata.v1 > weatherdata.v1.task.1.out
	-diff -w weatherdata.v1.task.1.out Output.expected/weatherdata.v1.task.1.out.expected


# Tests for Task II
unit2:
	python3 a1.task.2.unittest.v01.py NoDebug helpdata.v1 > helpdata.v1.task.2.out
	-diff -w helpdata.v1.task.2.out Output.expected/helpdata.v1.task.2.out.expected
	python3 a1.task.2.unittest.v01.py NoDebug weatherdata.v1 > weatherdata.v1.task.2.out
	-diff -w weatherdata.v1.task.2.out Output.expected/weatherdata.v1.task.2.out.expected


# Tests for Task III
unit3:
	python3 a1.task.3.unittest.v01.py NoDebug helpdata.v1 > helpdata.v1.task.3.out
	-diff -w helpdata.v1.task.3.out Output.expected/helpdata.v1.task.3.out.expected
	python3 a1.task.3.unittest.v01.py NoDebug weatherdata.v1 > weatherdata.v1.task.3.out
	-diff -w weatherdata.v1.task.3.out Output.expected/weatherdata.v1.task.3.out.expected


# Test for ooclassifierbase.py
base:
	python3 ooclassifierbase.py NoDebug helpdata.v1 > helpdata.v1.base.out
	-diff -w helpdata.v1.base.out Output.expected/helpdata.v1.base.out.expected
	python3 ooclassifierbase.py NoDebug weatherdata.v1 > weatherdata.v1.base.out
	-diff -w weatherdata.v1.base.out Output.expected/weatherdata.v1.base.out.expected


debug:
	python3 ooclassifierbase.py Debug helpdata.v1 > debug.helpdata.v1.base.out
	cat debug.helpdata.v1.base.out


clean:
	-rm helpdata.v1.base.out
	-rm weatherdata.v1.base.out
	-rm helpdata.v1.task.1.out
	-rm weatherdata.v1.task.1.out
	-rm helpdata.v1.task.2.out
	-rm weatherdata.v1.task.2.out
	-rm helpdata.v1.task.3.out
	-rm weatherdata.v1.task.3.out
	-rm helpdata.v1.alltasks.v01.out
	-rm weatherdata.v1.alltasks.v01.out
	-rm helpdata.v1.alltasks.v02.out
	-rm weatherdata.v1.alltasks.v02.out
	-/bin/rm -r -f Assn1.testcases.part.01
	-/bin/rm -r -f Assn1.testcases.part.02
	-/bin/rm -r -f Assn1.testcases.combined

copy:
	-/bin/rm -r -f Assn1.testcases.part.01
	mkdir -p Assn1.testcases.part.01/Output.expected
	cp Makefile ooclassifierbase.py helpdata.v1 weatherdata.v1 Assn1.testcases.part.01
	cp a1.task.*unit*py Assn1.testcases.part.01
	cp Output.expected/*expected Assn1.testcases.part.01/Output.expected
	tar cvfz assn1.testcases.part.01.tar.gz Assn1.testcases.part.01

copy2:
	-/bin/rm -r -f Assn1.testcases.part.02
	mkdir -p Assn1.testcases.part.02/Output.expected
	cp Makefile helpdata.v1 weatherdata.v1 Assn1.testcases.part.02
	cp a1.alltasks*py Assn1.testcases.part.02
	cp Output.expected/*alltasks**expected Assn1.testcases.part.02/Output.expected
	tar cvfz assn1.testcases.part.02.tar.gz Assn1.testcases.part.02

copyboth:
	-/bin/rm -r -f Assn1.testcases.combined
	mkdir -p Assn1.testcases.combined/Output.expected
	cp Makefile ooclassifierbase.py helpdata.v1 weatherdata.v1 Assn1.testcases.combined
	cp a1.*py Assn1.testcases.combined
	cp Output.expected/*expected Assn1.testcases.combined/Output.expected
	tar cvfz assn1.testcases.combined.tar.gz Assn1.testcases.combined


debug2:
	python3 a1.alltasks.v02.py Debug helpdata.v1 > helpdata.v1.alltasks.v02.out
	-diff -w helpdata.v1.alltasks.v02.out Output.expected/helpdata.v1.alltasks.v02.out.expected
	python3 a1.alltasks.v02.py Debug weatherdata.v1 > weatherdata.v1.alltasks.v02.out
	-diff -w weatherdata.v1.alltasks.v02.out Output.expected/weatherdata.v1.alltasks.v02.out.expected
	python3 a1.alltasks.v01.py Debug helpdata.v1 > helpdata.v1.alltasks.v01.out
	-diff -w helpdata.v1.alltasks.v01.out Output.expected/helpdata.v1.alltasks.v01.out.expected
	python3 a1.alltasks.v01.py Debug weatherdata.v1 > weatherdata.v1.alltasks.v01.out
	-diff -w weatherdata.v1.alltasks.v01.out Output.expected/weatherdata.v1.alltasks.v01.out.expected


# For Paul's use only, so use carefully
paul:
#	-cp helpdata.v1.base.out Output.expected/helpdata.v1.base.out.expected
#	-cp weatherdata.v1.base.out Output.expected/weatherdata.v1.base.out.expected
#	-cp helpdata.v1.task.1.out Output.expected/helpdata.v1.task.1.out.expected
#	-cp weatherdata.v1.task.1.out Output.expected/weatherdata.v1.task.1.out.expected
#	-cp weatherdata.v1.task.2.out Output.expected/weatherdata.v1.task.2.out.expected
#	-cp helpdata.v1.task.2.out Output.expected/helpdata.v1.task.2.out.expected
#	-cp weatherdata.v1.task.3.out Output.expected/weatherdata.v1.task.3.out.expected
#	-cp helpdata.v1.task.3.out Output.expected/helpdata.v1.task.3.out.expected
