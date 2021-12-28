# Chess Winrate Predictor

This was created as my final project for my Fall 2021 Mining Massive Datasets class at GMU. I used [Pyspark] and 
[Pyspark ML] to create a system to predict winrates of chess game states based on the large [Chess Games Dataset] on Kaggle. 

## Overview

The program takes data from the chess games dataset, which I first simplified by stripping unnecessary and irrelevant data and filtering for only standard chess games which ended not in a tie. Then, for each game, the list of moves in algebraic notation had to be translated to a list of states which occurred in the game. (Note: a 'state' as defined here regers to a chess diagram, as in just the information about what pieces are on the board and where). This requires a fair bit of disambiguation.

States were stored as strings, and translated into vectors for use with Pyspark ML. When initially using Linear Regression, mean error was around 0.3, which was disappointing. The issue was that since chess is so complex, around 90% of states only appeared in a small handful of games - under 20, even though the dataset had over 1 million games. This led to extreme win rates in over 85% of the original dataset (42% of gamestates had winrates under 0.1, and 45% had winrates over 0.9, both relative to the white player). Local-Sensitive Hashing (Minhashing) was used to combine similar states in the training set so that these winrates were less extreme. 

A more detailed report is available in the repository [here].
## Results

After implementing LSH, mean absolute error was reduced to around 9%. This is to say, the system predicted the winrate of a previously unknown chess state with an average of +-9%. I found this satisfactory given the limited time (about 3 weeks, since it was the end of the semester) and the raw complexity of the game of chess.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Chess Games Dataset]: <https://www.kaggle.com/mariuszmackowski/teamfight-tactics-fates-challenger-euw-rank-games>
   [Pyspark ML]: <https://spark.apache.org/docs/2.3.1/api/python/pyspark.ml.html>
   [Pyspark]: <https://spark.apache.org/docs/latest/api/python/index.html>
   [here]: (report.pdf)
   