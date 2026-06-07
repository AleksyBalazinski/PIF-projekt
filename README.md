# Project for the _Basic financial instruments_  ("Podstaowe instrumenty finansowe") course 

## Install required packages
```sh
pip install -r requirements.txt
```

## Data sources
* S&P 500 annual returns: [NYU Stern](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html)
* Inflation data: [World Bank Group](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=US)

## PDF report compilation
Run the notebook `analysis.ipynb` first to generate figures.
```sh
cp -r figures PIF-raport/
cd PIF-raport
pdflatex main.tex
```