# Tweet-classification
Estimating the location of tweet based only on the content of the tweet itself using a bag-of-words model.

The program accepts command line arguments like this:
./geolocate.py training-file testing-file output-file

The file format of the training and testing files is simple: one tweet per line, with the first word of the line indicating the actual location.
Output-file has the same format, except that the first word of each line is the estimated label, the second word is the actual label, and the rest of the line is the tweet itself.

The program also outputs (to the screen) the top 5 words associated with each of the 12 locations.
