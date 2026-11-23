.PHONY: all notes labs clean distclean

all: igmi.pdf igmi-labs.pdf

notes: igmi.pdf

labs: igmi-labs.pdf

igmi.pdf: igmi.tex preamble.tex $(wildcard */chap*.tex)
	latexmk -pdf -g -interaction=nonstopmode -halt-on-error igmi.tex

igmi-labs.pdf: igmi-labs.tex preamble.tex $(wildcard */lab*.tex)
	latexmk -pdf -g -interaction=nonstopmode -halt-on-error igmi-labs.tex

clean:
	latexmk -c igmi.tex
	latexmk -c igmi-labs.tex
	rm -f igmi.run.xml igmi.synctex.gz igmi-labs.run.xml igmi-labs.synctex.gz

distclean: clean
	rm -f igmi.pdf igmi-labs.pdf
