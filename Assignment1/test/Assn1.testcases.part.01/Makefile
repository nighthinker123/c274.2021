#### Assumes that ooclassifier.py is in the current working directory

all:
	make base
	make unit

unit:
	make unit1
	make unit2
	make unit3

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


clean:
	-rm helpdata.v1.base.out
	-rm weatherdata.v1.base.out
	-rm helpdata.v1.task.1.out
	-rm weatherdata.v1.task.1.out
	-rm helpdata.v1.task.2.out
	-rm weatherdata.v1.task.2.out
	-rm helpdata.v1.task.3.out
	-rm weatherdata.v1.task.3.out
	-/bin/rm -r -f Assn1.testcases.part.01

copy:
	-/bin/rm -r -f Assn1.testcases.part.01
	mkdir -p Assn1.testcases.part.01/Output.expected
	cp Makefile ooclassifierbase.py helpdata.v1 weatherdata.v1 Assn1.testcases.part.01
	cp a1.task.*unit*py Assn1.testcases.part.01
	cp Output.expected/*expected Assn1.testcases.part.01/Output.expected
	tar cvfz assn1.testcases.part.01.tar.gz Assn1.testcases.part.01


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
