README – Movie Reviews Data Analysis

Purpose of the Project
Tento projekt slouží k čištění, zpracování a analýze datasetu filmových recenzí uloženého v CSV souboru. Skript provádí kompletní datový pipeline:
- načtení dat,
- odstranění chybějících hodnot nebo jejich nastavení na defaultní,
- odstranění duplicit,
- čištění textových polí,
- převod datových typů,
- základní statistickou analýzu,
- filtrování a seskupování dat,
- vizualizace výsledků,
- uložení vyčištěného datasetu.

Cílem je získat přehled o kvalitě dat, odhalit trendy v recenzích a vytvořit vizuální výstupy pro další analýzu.

Dataset Source
Projekt očekává vstupní soubor:
data/original.csv

Dataset obsahuje typicky následující sloupce:
movie_title
review_text
rating
review_date
publish_date
reviewer
style
genre
review_length
word_count
sentiment_score
would_recommend
category
author

Skript automaticky kontroluje chybějící hodnoty, nekonzistence a datové typy.

Steps Performed

1. Initial Data Overview
Funkce DataOverview vypíše:
- prvních 10 řádků,
- počet řádků a sloupců,
- datové typy jednotlivých sloupců.

2. Missing Data Handling
Funkce MissingDataHandler:
- identifikuje řádky s chybějícími hodnotami,
- umožňuje buď smazat řádky s chybějícími daty, nebo doplnit defaultní hodnoty.

3. Duplicate Removal
Funkce DuplicateRemover:
- detekuje duplicitní řádky,
- umožňuje jejich odstranění.

4. Whitespace Cleaning
Funkce WhiteSpaceRemover:
- odstraňuje bílé znaky na začátku a konci textových polí.

5. Date Conversion
Skript převádí sloupce review_date a publish_date na datetime typ.

6. Statistical Analysis
Funkce BasicStatisticAnalysis:
- počítá průměr, medián, minimum, maximum a počet recenzí pro každý film,
- vytváří histogram hodnocení (1–10),
- počítá počet recenzí podle kategorie, autora a data publikace.

7. Filtering & Grouping
Funkce FilteringGrouping:
- vypisuje filmy s ratingem 8 a více,
- určuje 5 nejčastěji recenzovaných filmů,
- počítá počet recenzí v kategorii Tech,
- hledá nejaktivnějšího autora,
- počítá počet recenzí za jednotlivé měsíce.

8. Data Visualization
Funkce DataVisualization generuje grafy:
- průměrné hodnocení filmů (bar plot),
- počet recenzí v čase (line plot),
- počet recenzí podle kategorií (bar plot),
- počet recenzí měsíčně (line plot),
- histogram délky recenze.

Grafy se ukládají do složky:
plots/

9. Saving Cleaned Data
Vyčištěný dataset se uloží jako:
data/cleaned_data.csv

Sample Insights
Z datasetu lze získat například:
- které filmy mají nejvyšší průměrné hodnocení,
- které dny mají nejvíce publikovaných recenzí,
- jaké kategorie jsou nejčastěji recenzované,
- kdo je nejaktivnější recenzent,
- jak se počet recenzí mění v čase,
- jak dlouhé jsou recenze a jak se liší mezi kategoriemi.

How to Run the Script

1. Install dependencies
Projekt používá:
pandas
matplotlib

Instalace:
pip install pandas matplotlib

2. Prepare folder structure
project/
  data/
    original.csv
  plots/
  script.py

Složka plots musí existovat:
mkdir plots

3. Run the script
python script.py

Skript načte data, provede čištění, vytvoří grafy a uloží vyčištěný dataset.
