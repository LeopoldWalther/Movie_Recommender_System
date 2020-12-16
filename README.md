# Movie_Recommender_System
A Movie Recommender system developed during the Udacity Nanodegree Data Scientist. The Recommender uses a hybrid of collaborative filtering based recommendations, knowledge based recommendations and content based recommendations.

### Table of Contents

1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions: <a name="instructions"></a>

The project was created with Python 3.8.0.
Run the following commands to initiate thw project:

1. create virtual environment:

  `python3 -m venv movie_recommender_env`

2. activate the virtual environment:

  `source movie_recommender_env/bin/activate`

3. pip install required packages:

  `pip install -r requirements.txt`


## Project Motivation: <a name="motivation"></a>

Motivation of this project is to implement a recommender system 
including knowledge based recommendations, collaborative filterung based
recommendations and content based recommendations using Python.
The data used is downloaded from
[Movie Tweetings Data](https://github.com/sidooms/MovieTweetings).


## File Descriptions: <a name="files"></a>
The project contains the following files:

```
Movie_Recommender_System/
│
├── README.md
├── requirements.txt
├── recommender.py
├── recommender_functions.py
├── data/
│   ├── movies_clean.csv  # movie data already cleaned
│   ├── train_data  # review data already prepared for training the model

```


## Licensing, Authors, Acknowledgements: <a name="licensing"></a>

The data used is downloaded from
[Movie Tweetings Data](https://github.com/sidooms/MovieTweetings).

Feel free to use my code as you please:

Copyright 2020 Leopold Walther

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
