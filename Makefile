JOBNAME = igmi

.PHONY: all clean distclean

all: $(JOBNAME).pdf

$(JOBNAME).pdf: main.tex preamble.tex $(wildcard */notes.tex)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=$(JOBNAME) main.tex

clean:
	latexmk -c -jobname=$(JOBNAME) main.tex
	rm -f $(JOBNAME).run.xml $(JOBNAME).synctex.gz

distclean: clean
	rm -f $(JOBNAME).pdf
