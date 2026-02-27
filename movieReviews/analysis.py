import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt


def WhiteSpaceRemover(DataFrame):
    """
      Funkce pro odstraneni bilych znaku ze stringu ve vsech sloupecich
        Input:
            DataFrame: DataFrame, ktery chceme upravit
        Output
            DataFrame, ktery byl upraven
    """
    DataFrame = DataFrame.copy() #Bezpecnejsi, aby se neupravoval puvodni DataFrame, ale jen kopie
    DataFrame = DataFrame.apply(lambda col: col.str.strip() if col.dtype == "str" else col)
    #DataFrame.apply() -> pouzije kazdy sloupec z DataFrame
    #lambda col: -> anonymni funce ktera pro sloupec col provede nasledujici logiku
    #col.str.strip() -> odstrani bile znaky z leveho a praveho konce stringu
    #col.dtype == "str" -> pouzije se jen pokud sloupec je typu string
    return DataFrame #Navrati upraveneho DataFrame

def DuplicateRemover(DataFrame):
    """
      Funkce pro odstraneni duplicitnich radku
        Input:
            DataFrame: DataFrame, ktery chceme upravit
        Output
            DataFrame, ktery byl upraven
    """
    # @todo conside differentiation of input and output DataFrame names to make it clearer
    DataFrame = DataFrame.copy() #Bezpecnejsi, aby se neupravoval puvodni DataFrame, ale jen kopie
    if(DataFrame.duplicated().sum()) > 0: #Overeni ze existuji duplicitni radky
        print("\nNa techto radcich je duplicita")
        print(DataFrame[DataFrame.duplicated()])
    # @todo consider moving input outside of this function, to make it more reusable and testable
        if(input("Chces smazat duplicitni radky? y/n: ") == "y"):
            DataFrame = DataFrame.drop_duplicates() #smazani
            print("Duplicita odstranena")
        else:
            print("Duplicita zustane zachovana")
    #else:
    #  print("Soubor neobsahuje duplicitu, vse OK")
    return DataFrame #Navrati upraveneho DataFrame


def MissingDataHandler(DataFrame):
    """
      Funkce pro osetreni chybějících dat
        Input:
            DataFrame: DataFrame, ktery chceme upravit
        Output
            DataFrame, ktery byl upraven
    """
    DataFrame = DataFrame.copy() #Bezpecnejsi, aby se neupravoval puvodni DataFrame, ale jen kopie
    if(DataFrame.isnull().any().any() == True): #neco, nekde chybi
        print("Nektera data v souboru chybi\nVypis chybejich:")
        #print(DataFrame.isnull().sum()) #soucet chybejich hodnot pro kazde sloupecky zvlast
        print(DataFrame[DataFrame.isnull().any(axis=1)]) #vypise radky kde neco chybi
        print("Jak chcete data upravit?")
        print("1 - smazat radky s chybejicimi daty")
        print("2 - doplnit na defaultni hodnoty")
        chose = int(input("Zadej volbu jak data opravit: "))
        # @todo consider moving input outside of this function, to make it more reusable and testable
        # @todo consider addiing addtional function parameter choose 
        if(chose == 1): #smazani radku s chybejicimi daty
            #DataFrame = DataFrame.dropna() #smaze vsechny radky kde neco chybi 
            #!!!! TOTO MI NEFUNGUJE, smaze to cely DataFrame a neprisel jsem na to proc, udajne to je tim, ze tam jsou skryte NaN hodnoty
    
            DataFrame = DataFrame.dropna(subset=["movie_title", "review_text", "rating", "review_date"
                                                 , "reviewer", "style", "genre", "review_length", "word_count"
                                                 , "sentiment_score", "would_recommend"]) #smaze jen radky, kde neco chybi v techto sloupeccich
            print("Data smazana")
        elif(chose == 2): #doplneni na defaultni hodnoty
            #toto projde kazdy sloupecek a doplni tam defaultni hodnotu, pokud tam neco chybi, ale jen kdyz tam neco chybi!
            # @todo consider getting rid off magic values and moving them to constants, to make it more maintainable and readable
            DataFrame["movie_title"] = DataFrame["movie_title"].fillna("Unknown")
            DataFrame["review_text"] = DataFrame["review_text"].fillna("No review")
            DataFrame["rating"] = DataFrame["rating"].fillna(0)
            DataFrame["review_date"] = DataFrame["review_date"].fillna("1900-01-01") #defaultni datum, ktery se pouzije, pokud tam neco chybi
            DataFrame["reviewer"] = DataFrame["reviewer"].fillna("Unknown")
            DataFrame["style"] = DataFrame["style"].fillna("")
            DataFrame["genre"] = DataFrame["genre"].fillna("")
            DataFrame["review_length"] = DataFrame["review_length"].fillna(0)
            DataFrame["word_count"] = DataFrame["word_count"].fillna(0)
            DataFrame["sentiment_score"] = DataFrame["sentiment_score"].fillna(0)
            DataFrame["would_recommend"] = DataFrame["would_recommend"].fillna(False)
            print("Data doplnena na defaultni hodnoty")
    #else:
    #  print("Soubor neobsahuje chybejici data, vse OK")
    return DataFrame #Navrati upraveneho DataFrame


def DataOverview(DataFrame):
    """
      Funkce pro prvni vizualizaci dat v CSV
        Input:
          DataFrame: DataFrame, ktery chceme vizualizovat
        Output
          none
    """
    print("Prvnich 10 radku souboru")
    print(DataFrame.head(10)) #vypise prvnich 10 radku
    rows, columns = DataFrame.shape #shape je tvar datasetu = proste kolik ma DataFrame radku s sloupcu
    print(f"\nSoubor ma {rows} radku\nSoubor ma {columns} sloupecku")
    print("\nHlavicka souboru a datove typy jednotlivych sloupcu:")
    DataFrame.info() #toto vypise hlavicku a datove typy jednotlivych sloupcu, takze i s indexem
    #info() nepotrebuje print, samo se vypise

def BasicStatisticAnalysis(DataFrame):
    """
      Funkce pro statistickou analyzu dat
        Input:
          DataFrame: DataFrame, ktery chceme analyzovat
        Output
          none
    """
    #@todo better meaningful comment
    #Zobrazeni zakladnich statistik pro sloupecek rating
    grouped = DataFrame.groupby("movie_title")["rating"] #toto jen informuje podle ceho se grupuje a co nas bude zajimat, pak to nemusim vsude psat

    MovieFrame = pd.DataFrame({
        "mean": grouped.mean().round(2),
        "median": grouped.median().round(2),
        "min": grouped.min().round(2),
        "max": grouped.max().round(2),
        "reviews": grouped.size()
    }).reset_index()
    print("\nHodnoceni filmu dle ratingu")
    print(MovieFrame.to_string(index=False)) #vypise hodnoceni

    # Vytvoreni histogramu pro sloupecek rating
    all_ratings = pd.Series(range(1, 11), name="rating") #toto vytvori serii 1-10 a pouzije se jako osa X, pokud by nejaky rating chybel, tak na ose bude s kodnotou 0
    counts = DataFrame["rating"].value_counts().reindex(all_ratings, fill_value=0)
    #DataFrame["rating"].value_counts() --> toto spocita kolikrat se kazda hodnota vyskytuje, ale sezadi to dle cetnosti
    #.reindex(all_ratings, fill_value=0) --> toto presklada vysledky dle indexu 1-10 a pokud neco chybi tak zada 0

    plt.figure(figsize=(8,5)) #graf bude 8 palcu siroky a 5 palcu vysoky
    counts.plot(kind="bar", color="green") #toto vytvori bar graph, kde osa X bude rating a osa Y bude pocet vyskytu, barva bude zelena

    plt.xlabel("Rating filmu")
    plt.ylabel("Počet výskytů")
    plt.title("Histogram hodnocení (1-10)")
    plt.xticks(rotation=0)
    plt.savefig("plots/rating_histogram.png", dpi=300, bbox_inches="tight") #bbox_inches="tight" zajisti, ze se graf neodreze
    #plt.show() #graf jen ulozim, videt ho nechci

    #Pocet hodnoceni pro kazdou kategorii
    print("\nPocet review pro kazdou kategorii")
    counts = DataFrame.groupby("category").size().reset_index(name="count") #secte kategorie, chci data frame = popsane hlavicky sloupecku
    print(counts.to_string(index=False)) #do vypisu ale nechci videt index
    #Pocet hodnoceni od kazdeho autora
    print("\nPocet review od kazdeho autora")
    counts = DataFrame.groupby("author").size().reset_index(name="count") #secte autory, chci data frame = popsane hlavicky sloupecku
    print(counts.to_string(index=False)) #do vypisu ale nechci videt index
    #Nejcastejsi den kdy se publikovalo
    print("\nDen kdy bylo nejvice review")
    counts = DataFrame.groupby("review_date").size().reset_index(name="count") #secte kdz se delalo review, chci data frame = popsane hlavicky sloupecku
    top_day = counts.sort_values(by="count", ascending=False).head(1) #seradi podle count a vezme jen ten nejvyssi
    print(top_day.to_string(index=False)) #do vypisu ale nechci videt index

    #Summary statistics for numeric columns (e.g., views)
    #TODO
    #Co se tady ocekava?


def FilteringGrouping(DataFrame):
    """
      Funkce pro filtrovani a slucovani dat
        Input:
          DataFrame: DataFrame, ktery chceme analyzovat
        Output:
          None
    """
    #Seznam filmu s hodnocenim 8 a vice
    unique_movies = (
        DataFrame.loc[DataFrame["rating"] >= 8, "movie_title"] #.loc musi byt pokud chci filtrovat a pak vybrat jeste konkretni sloupec
        .drop_duplicates() #odstrani duplicitni nazvy filmu
        .reset_index(drop=True)
        .sort_values() #seradi nazvy filmu podle abecedy
         )
    print("\nFilmy s hodnocenim 8 a vice:")
    print(unique_movies.to_string(index=False))

    #5 nejvice recenzovanych filmu
    print("\n5 nejvice recenzovanych filmu")
    counts = DataFrame.groupby("movie_title").size().reset_index(name="count") #secte filmy, chci data frame = popsane hlavicky sloupecku
    movie_review = counts.sort_values(by="count", ascending=False).head(5) #seradi podle count a vezme jen ten nejvyssi
    print(movie_review.to_string(index=False)) #do vypisu ale nechi videt index

    #pocet review v kategorii Technology
    print("\nPocet review v kategorii Technology (Tech)")
    counts = (DataFrame["category"] == "Tech").sum()
    print(counts) #do vypisu ale nechi videt index
    
    #nejvice aktivni autor review
    print("\nNejvice aktivni autor review")
    counts = DataFrame.groupby("reviewer").size().reset_index(name="count") #secte pocet review
    reviewer = counts.sort_values(by="count", ascending=False).head(1) #seradi podle count a vezme jen ten nejvyssi
    print(reviewer.to_string(index=False)) #do vypisu ale nechi videt index

    #pocet clanku publikovanych kazdy mesic
    DataFrame["review_month"] = DataFrame["review_date"].dt.to_period("M") #sloupecek si prejmenuji na review_month
    review_month = DataFrame.groupby("review_month").size().reset_index(name="count")
    print(f"\nPocet review v kazdem mesici: \n{review_month.to_string(index=False)}") #do vypisu ale nechi videt index


def DataVisualization(DataFrame):
    """
    Visualizace dat
      Input:
        DataFrame: DataFrame, ktery chceme analyzovat
      Output:
        None
    """
    #Rating histogram --> udelano uz v BasicStatisticAnalysis
    
    #prumerne hodnoceni kazdeho filmu - bar plot
    MovieMeanRating = DataFrame.groupby("movie_title")["rating"].mean()

    plt.figure(figsize=(10,5)) #graf bude 8 palcu siroky a 5 palcu vysoky
    MovieMeanRating.plot(kind="bar", color="green") #toto vytvori bar graph, kde osa X bude rating a osa Y bude pocet vyskytu, barva bude zelena

    plt.xlabel("Jmeno filmu")
    plt.ylabel("Prumerny rating")
    plt.title("Prumerne hodnoceni kazdeho filmu")
    plt.xticks(rotation=90)
    plt.savefig("plots/average_move_rating.png", dpi=300, bbox_inches="tight") #bbox_inches="tight" zajisti, ze se graf neodreze
    #plt.show() #graf jen ulozim, videt ho nechci

    #pocer review v case - line plot
    #prumerne hodnoceni kazdeho filmu - bar plot
    ReviewsPerDate = DataFrame.groupby("review_date").size() #secte kdy se delalo review
    plt.figure(figsize=(10,5)) #graf bude 10 palcu siroky a 5 palcu vysoky
    ReviewsPerDate.plot(kind="line", color="green") #toto vytvori bar graph, kde osa X bude rating a osa Y bude pocet vyskytu, barva bude zelena
    plt.xlabel("Datum")
    plt.ylabel("Pocet review")
    plt.title("Pocet review v case")

    #nastaveni aby osa Y zobrazovala jen cela cisla, protoze pocet review nemuze byt desetina nebo neco podobneho
    import matplotlib.ticker as mticker
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.xticks(rotation=0)
    plt.savefig("plots/reviews_in_time.png", dpi=300, bbox_inches="tight") #bbox_inches="tight" zajisti, ze se graf neodreze
    #plt.show() #graf jen ulozim, videt ho nechci

    #pocet clanku pro kazdou kategorii - bar plot
    ReviewsPerCategory = DataFrame.groupby("category").size() #secte kategorie
    plt.figure(figsize=(8,5)) #graf bude 8 palcu siroky a 5 palcu vysoky
    ReviewsPerCategory.plot(kind="bar", color="green") #toto vytvori bar graph
    plt.xlabel("Kategorie")
    plt.ylabel("Pocet review")
    plt.title("Pocet review pro kazdou kategorii")
    plt.xticks(rotation=0) 
    plt.savefig("plots/reviews_per_category.png", dpi=300, bbox_inches="tight") #bbox_inches="tight" zajisti, ze se graf neodreze

    #pocet clanku mesicne - line plot
    ReviewsPerMonth = DataFrame.groupby("review_month").size() #secte kdy se delalo review
    plt.figure(figsize=(10,5)) #graf bude 10 palcu siroky a 5 palcu vysoky
    ReviewsPerMonth.plot(kind="line", color="green") #toto vytvori
    plt.xlabel("Mesic")
    plt.ylabel("Pocet review")
    plt.title("Pocet review mesicne")
    plt.xticks(rotation=90)
    plt.savefig("plots/reviews_per_month.png", dpi=300, bbox_inches="tight") #bbox_inches="tight" zajisti, ze se graf neodreze

    #histogram delky recenze - histogram
    plt.figure(figsize=(8,5)) #graf bude 8 palcu siroky a 5 palcu vysoky
    DataFrame["review_length"].plot(kind="hist", bins=20, color="green") #toto vytvori histogram, kde osa X bude delka recenze a osa Y bude pocet vyskytu, barva bude zelena
    plt.xlabel("Delka recenze") 
    plt.ylabel("Pocet review") 
    plt.title("Histogram delky recenze")
    plt.xticks(rotation=0)
    plt.savefig("plots/review_length_histogram.png", dpi=300, bbox_inches="tight") #bbox_inches="tight" zajisti, ze se graf neodreze



# *************************************
# Hlavni analyza
# *************************************
#Nacteni CSV
DataFrame = pd.read_csv("data/original.csv", sep = ",")
if DataFrame.columns.isnull().any(): #overeni ze hlavicka neobsahuje prazdne hodnoty
    print("Některé názvy sloupců chybí!")
#tady pozor, Dusanovo CSV ma v bunkach taky carky, ale v uvozovkach. Ty se pak neuvazuji jako oddelovac

#Prvni vizualizaci dat v CSV
DataOverview(DataFrame)

#Overeni a osetreni chybějících dat
DataFrame = MissingDataHandler(DataFrame) 

#Overeni a odstraneni duplicit
DataFrame = DuplicateRemover(DataFrame)

#Odstraneni bilych znaku ze stringu ve vsech sloupcich
DataFrame = WhiteSpaceRemover(DataFrame)

#prevod datumu (sloupecek review_date a publish_date) ze stringu na format datum
DataFrame["review_date"] = pd.to_datetime(DataFrame["review_date"])
DataFrame["publish_date"] = pd.to_datetime(DataFrame["publish_date"])

#Zakladni statisticka analyza dat
BasicStatisticAnalysis(DataFrame)

#Filtering and Grouping
FilteringGrouping(DataFrame)

#Visualizace dat
DataVisualization(DataFrame)


#Konec, ukladam upravena data
DataFrame.to_csv("data/cleaned_data.csv", sep=",", index=False)
print("\nUpravena data byla ulozena do souboru cleaned_data.csv")
