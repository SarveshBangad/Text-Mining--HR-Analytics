#Text Mining Case Study:
#----Loading all Required Packages---------------------------------
library("qdap")
library("tm")
library("wordcloud")
library("dendextend")
library("RColorBrewer")
library("plotrix")
library("RWeka")
library("readr")

#------1. Load readr package and Read csv files of Amazon and Google


url_amzn_csv <- "https://assets.datacamp.com/production/course_935/datasets/500_amzn.csv"
url_goog_csv <- "https://assets.datacamp.com/production/course_935/datasets/500_goog.csv"
amzn <- read_csv(url_amzn_csv)
goog <- read_csv(url_goog_csv)


#amzn_csv <- read_csv("G:/BAPM/Technology/DataCamp/R Track/Sentiment Analysis/1. Text Mining - Bag of Words Approach/DataSet/amazon.csv", col_types = c("numeric","character","character","character"))
#google_csv <- read_csv("G:/BAPM/Technology/DataCamp/R Track/Sentiment Analysis/1. Text Mining - Bag of Words Approach/DataSet/google.csv")

#amzn <- amzn_csv
#goog <- google_csv


#rm(amzn_csv)
#rm(amzn_csv1)
#rm(google_csv)


#amzn_csv1 <- read_delim("G:/BAPM/Technology/DataCamp/R Track/Sentiment Analysis/1. Text Mining - Bag of Words Approach/DataSet/amazon.csv", delim = ",")

#-----2. Examine the text sources-------
# Print the structure of amzn
str(amzn)

# Create amzn_pros
amzn_pros <- amzn$pros

# Create amzn_cons
amzn_cons <- amzn$cons

# Print the structure of goog
str(goog)

# Create goog_pros
goog_pros <- goog$pros


# Create goog_cons
goog_cons <- goog$cons

#---------3. Text Organization-------

#qdap Cleaning function
qdap_clean <- function(x){
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <- replace_ordinal(x)
  x <- replace_symbol(x)
  x <- tolower(x)
  return(x)
}

# tm cleaning function
tm_clean <- function(corpus){
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"),"Google","Amazon","company"))
  return(corpus)
}


#Clean up Amazon Reviews
# Alter amzn_pros
amzn_pros <- qdap_clean(amzn_pros)
amzn_pros <- na.omit(amzn_pros)

# Alter amzn_cons
amzn_cons <- qdap_clean(amzn_cons)
amzn_cons <- na.omit(amzn_cons)

# Create az_p_corp 
az_p_corp <- VCorpus(VectorSource(amzn_pros))

# Create az_c_corp
az_c_corp <- VCorpus(VectorSource(amzn_cons))

# Create amzn_pros_corp
amzn_pros_corp <- tm_clean(az_p_corp)

# Create amzn_cons_corp
amzn_cons_corp <- tm_clean(az_c_corp)

# Clean up Google Reviews
# Apply qdap_clean to goog_pros
goog_pros <- qdap_clean(goog_pros)
goog_pros <- na.omit(goog_pros)

# Apply qdap_clean to goog_cons
goog_cons <- qdap_clean(goog_cons)
goog_cons <- na.omit(goog_cons)

# Create goog_p_corp
goog_p_corp <- VCorpus(VectorSource(goog_pros))

# Create goog_c_corp
goog_c_corp <- VCorpus(VectorSource(goog_cons))

# Create goog_pros_corp
goog_pros_corp <- tm_clean(goog_p_corp)

# Create goog_cons_corp
goog_cons_corp <- tm_clean(goog_c_corp)


#---4 & 5. Feature Extraction and Analysis----

#define tokenizer function to create bigrams:
tokenizer <- function(x)
  NGramTokenizer(x, Weka_control(min = 2, max = 2))

#--Word Cloud of Amazon Pros Comment---
# Create amzn_p_tdm
amzn_p_tdm <- TermDocumentMatrix(amzn_pros_corp, control = list(tokenize = tokenizer))

# Create amzn_p_tdm_m
amzn_p_tdm_m <- as.matrix(amzn_p_tdm)


# Create amzn_p_freq
amzn_p_freq <- rowSums(amzn_p_tdm_m)

# Plot a wordcloud using amzn_p_freq values
wordcloud(names(amzn_p_freq),amzn_p_freq, max.words = 25, color = "blue")


#--Word Cloud of Amazon Cons Comment---
# Create amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(amzn_cons_corp, control = list(tokenize = tokenizer))

# Create amzn_c_tdm_m
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)

# Create amzn_c_freq 
amzn_c_freq <- rowSums(amzn_c_tdm_m)

# Plot a wordcloud of negative Amazon bigrams
wordcloud(names(amzn_c_freq), amzn_c_freq, max.words = 25, color = "red")


#----Dendrogram for Amazon Cons-------------------

# Create amzn_c_tdm2 by removing sparse terms 
amzn_c_tdm2 <- removeSparseTerms(amzn_c_tdm , sparse = 0.993)

# Create hc as a cluster of distance values
hc <- hclust(dist(amzn_c_tdm2, method = "euclidean"), method = "complete")

# Produce a plot of hc
plot(hc)


#----Word Association for Amazon Pros--------

# Create term_frequency for Amazon Pros Reviews sorted in descending order
amz_p_term_frequency <- sort(amzn_p_freq , decreasing = TRUE)
 
# Print the 5 most common terms
print(amz_p_term_frequency[1:5])

# Find associations with fast paced
findAssocs(amzn_p_tdm,"fast paced", 0.2)


#----Comparison Cloud for Google Pros and Cons Reviews------
all_pros_goog <- paste(goog_pros, collapse = " ")
all_cons_goog <- paste(goog_cons, collapse = " ")

all_goog_reviews <- c(all_pros_goog,all_cons_goog)

# Create Corpus and Clean all google reviews:
all_goog_reviews_source <- VectorSource(all_goog_reviews)
all_goog_corp <- VCorpus(all_goog_reviews_source)
all_goog_corp <- tm_clean(all_goog_corp)

# Create TDM of all Google Reviews
all_goog_tdm <- TermDocumentMatrix(all_goog_corp)

# Name the columns of all_goog_tdm
colnames(all_goog_tdm) <- c("Goog_Pros", "Goog_Cons")


# Create all_goog_m
all_goog_m <- as.matrix(all_goog_tdm)

# Build a comparison cloud
comparison.cloud(all_goog_m, colors = c("#F44336", "#2196f3"), max.words = 100)


#------Pyramid Cloud for Pros of Amazon and Google Reviews-----

all_pros_amzn <- paste(amzn_pros, collapse = " ")
all_pros_goog <- paste(goog_pros, collapse = " ")

all_pros_reviews <- c(all_pros_amzn,all_pros_goog)

# Create Corpus and Clean all google and amazon pros reviews:
all_pros_reviews_source <- VectorSource(all_pros_reviews)
all_pros_corp <- VCorpus(all_pros_reviews_source)
all_pros_corp <- tm_clean(all_pros_corp)

# Create TDM of all Google and Amazon pros Reviews
all_pros_tdm <- TermDocumentMatrix(all_pros_corp, control = list(tokenize = tokenizer))

# Name the columns of all_pros_tdm
colnames(all_pros_tdm) <- c("Amazon Pro", "Google Pro")


# Create all_pros_tdm_m
all_pros_tdm_m <- as.matrix(all_pros_tdm)

head(all_pros_tdm_m)

# Create common_words
common_words <- subset(all_pros_tdm_m, all_pros_tdm_m[, 1] > 0 & all_pros_tdm_m[, 2] > 0)

# Create difference
difference <- abs(common_words[,1] - common_words[,2] )

# Add difference to common_words
common_words <- cbind(common_words,difference)

# Order the data frame from most differences to least
common_words <- common_words[order(common_words[, 3], decreasing = TRUE),]

# Create top15_df
top15_df <- data.frame(x = common_words[1:15, 1], 
                       y = common_words[1:15, 2], 
                       labels = rownames(common_words[1:15, ]))

# Create the pyramid plot
pyramid.plot(top15_df$x, top15_df$y, labels = top15_df$labels, gap = 12, 
             top.labels = c("Amzn", "Pro Words", "Google"), 
             main = "Words in Common", unit = NULL)



#------Pyramid Cloud for Cons of Amazon and Google Reviews-----

all_cons_amzn <- paste(amzn_cons, collapse = " ")
all_cons_goog <- paste(goog_cons, collapse = " ")

all_cons_reviews <- c(all_cons_amzn,all_cons_goog)

# Create Corpus and Clean all google and amazon pros reviews:
all_cons_reviews_source <- VectorSource(all_cons_reviews)
all_cons_corp <- VCorpus(all_cons_reviews_source)
all_cons_corp <- tm_clean(all_cons_corp)

# Create TDM of all Google and Amazon pros Reviews
all_cons_tdm <- TermDocumentMatrix(all_cons_corp, control = list(tokenize = tokenizer))

# Name the columns of all_pros_tdm
colnames(all_cons_tdm) <- c("Amazon Con", "Google Con")


# Create all_pros_tdm_m
all_cons_tdm_m <- as.matrix(all_cons_tdm)

head(all_cons_tdm_m)

# Create common_words
common_words_cons <- subset(all_cons_tdm_m, all_cons_tdm_m[, 1] > 0 & all_cons_tdm_m[, 2] > 0)

# Create difference
difference_cons <- abs(common_words_cons[,1] - common_words_cons[,2] )

# Add difference to common_words
common_words_cons <- cbind(common_words_cons,difference_cons)

# Order the data frame from most differences to least
common_words_cons <- common_words_cons[order(common_words_cons[, 3], decreasing = TRUE),]

# Create top15_df
top15_df_cons <- data.frame(x = common_words_cons[1:15, 1], 
                       y = common_words_cons[1:15, 2], 
                       labels = rownames(common_words_cons[1:15, ]))

# Create the pyramid plot
pyramid.plot(top15_df_cons$x, top15_df_cons$y, labels = top15_df_cons$labels, gap = 12, 
             top.labels = c("Amzn", "Con Words", "Google"), 
             main = "Words in Common", unit = NULL)


