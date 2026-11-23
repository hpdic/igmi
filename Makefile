NOTES_JOB = igmi
LABS_JOB  = igmi-labs

.PHONY: all notes labs clean distclean

all: $(NOTES_JOB).pdf $(LABS_JOB).pdf

notes: $(NOTES_JOB).pdf

labs: $(LABS_JOB).pdf

$(NOTES_JOB).pdf: main.tex preamble.tex $(wildcard */chap*.tex)
	latexmk -pdf -g -interaction=nonstopmode -halt-on-error -jobname=$(NOTES_JOB) main.tex

$(LABS_JOB).pdf: labs.tex preamble.tex $(wildcard */lab*.tex)
	latexmk -pdf -g -interaction=nonstopmode -halt-on-error -jobname=$(LABS_JOB) labs.tex

clean:
	latexmk -c -jobname=$(NOTES_JOB) main.tex
	latexmk -c -jobname=$(LABS_JOB) labs.tex
	rm -f $(NOTES_JOB).run.xml $(NOTES_JOB).synctex.gz $(LABS_JOB).run.xml $(LABS_JOB).synctex.gz

distclean: clean
	rm -f $(NOTES_JOB).pdf $(LABS_JOB).pdf
