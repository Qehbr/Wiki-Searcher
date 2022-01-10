# Wiki-Searcher
Wiki search engine
File explanation and how to use it:


1. inverted_index.py
Inverted inde for body, titles, anchors of wiki pages. Also, contains MultiFileWriter and MultiFileReader classes, writing/reading to/from GCP bucket all postings and postings locations from current inverted index

2. crawl_and_index.ipynb 
Contains all crawling and indexing code. 
Firstly, it gets all wikidata from wikidumps. 
Then, it creates indexes for body, title (stemmed using Porter Stemmer), anchor. 
Writes all postings and postings locations (using MultiFileWriter) to GCP, and all inverted indexes data (globals) to GCP. 
Calculates PageRank and uploads it to GCP to JSON file. Downloads Page View to JSON file and uploads it to GCP. 
Makes id, title JSON file and uploads it to GCP. 

3. ranking.py 
Contains all relevant functions to TF-IDF for body searching

4. search_frontend_measures.py
Allowes to test different weights on (title, body, anchor, page rank, page view) and see how different measurements (MAP, Recall, Precision, R-Precision) changes respectively

5. search_frontend.py
Main script, flask app, containing all searching logic
