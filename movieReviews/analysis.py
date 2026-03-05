import pandas as pd
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))                     #find the root directory where the script is located
original_file_path = script_dir + "\\data\\original_movie_reviews_copy.csv"      #make path for the original csv file
cleand_file_path = script_dir + "\\data\\cleaned_movie_reviews.csv"
original_file = pd.read_csv(original_file_path)

#Display:
print("This is first 10 rows in the CSV")
print(original_file[0:9])
print("Dataset shape is: ", original_file.shape)
print("Names of the columns are: ", original_file.columns)
print("Data types using info(): ")
print(original_file.info())



#Identify:
print("Řádky s prázdnýma hodnotama:")
print(original_file.isna())
print(original_file[original_file.isna().any(axis=1)])      # print missing rows
print("Duplikované řádky: ")
print(original_file[original_file.duplicated(keep=False)])  #print duplicated rows

#Cleaning the data
original_file.drop_duplicates(inplace = True)   #remove duplicates in the original data frame
original_file.dropna(inplace = True)    #remove rows with empty cells in the original data frame

original_file["review_date"] = pd.to_datetime(original_file["review_date"]) #convert the date into datetime64 data type

for series in original_file:
    if original_file[series].dtype == "str":
        original_file[series] = original_file[series].str.replace(" ","") #delete all spacebars in strings
        original_file[series] = original_file[series].str.lower() #normalize to lower case strings

original_file.to_csv(cleand_file_path,index = False)

#Basic statistical analysis
print(original_file["rating"].mean())
print(original_file["rating"].median())
print(original_file["rating"].min())
print(original_file["rating"].max()) 

print(original_file["movie_title"].value_counts())

#filtering and grouping
ratingBiggerEight = original_file[original_file["rating"] >= 8]
print(ratingBiggerEight)
topFiveReviewed = original_file["movie_title"].value_counts().head(5)
print(topFiveReviewed)
averageRating = original_file.groupby("movie_title")["rating"].mean()
print(averageRating)

#data visualization
plt.hist(original_file["rating"],   color = "blue",
                                    edgecolor = "black")
plt.title("Distribution of ratings")
plt.show()

plt.barh(averageRating.index,averageRating.values,  color = "blue",
                                                    edgecolor = "black")
plt.subplots_adjust(left=0.25)
plt.title("Average rating per movie")
plt.show()