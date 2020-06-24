# Retail-Store-Location-Ranking
Select the best location for a new opening retail store by ranking. 

Retail store location selection is an important task for retailers and investors. For an new opening retail store, business analyst and data scientists analyze the local area, number of potential costumers, competitors, cost and make decision by ranking the candidate store location.

In this project, we exploit data analysis and machine learning approach to help predict the ranking of retail store location candidates. The aim is to find elements that benefit the retail quality of a location, and build predictive model to decide highest ranking candidate for a new opening retail store.

Data Collection: 
The retail store sample we choose to analyze in this project is 50 Mcdonald’s restaurants in New York. The data is collected by Foursquare API, and the target store popularity is collected by total comments from Google Place API. We selected New York city as our analyze target because there are more API user locate in the city, and more data we could collect around that area. Originating from the geographic center of New York, we collect data of all Mcdonald’s from a circle range of 15 kilometer radius. Due to the limit of query of Google Place API, totally we get 50 retail stores with there total comments. We assumed total comments on Google Map represents each store’s popularity.

![image](https://github.com/laurence-lin/Retail-Store-Location-Ranking/blob/master/area.jpg)
Fig. 1 The area to analyze in New York. Red spot is the location of retail store

![image](https://github.com/laurence-lin/Retail-Store-Location-Ranking/blob/master/retail_store.jpg)
Fig. 2 The nearby area of a retail store to analze

For each retail store, we collect geographic data such as neighbor competitors. The geographic data is collected nearby each retail store within 200 meter radius. The features includes:

Density: number of neighbor venues

Competitiveness: number of same type (fast food restaurant) in the nearby area

Neighbor entropy: the diversity of venue types in the nearby area

Total competitiveness: number of all possible competitors (all type restaurants) in the nearby area

Area popularity: Number of residential venue in the nearby area

We summarize the factors that influence a retail store’s popularity: Consumer behavior, area popularity, and number of competitors. By Foursquare API, we scrape density and residential venues as area popularity, and competitiveness for fast food restaurant and all type restaurants. Space heterogeneity is considered by previous research, that diversity of venue implies different type of potential consumers.
Exploratory Data Analysis: We observe the relationship between each features and the target, number of comments.

Fig. 3 Distribution of each feature with respect to target
We’ve found that it’s hardly to discover linear relationship between each feature and the target. However, by the visualization we could found two outliers in the data, we could drop these two samples for data cleaning.
Then, we could observe the relationship between features, there may be some hidden information to explore:

Fig. 4 Residence venue VS. Density
Residential venues and density have positive relationship. It’s reasonable since residential venue is contained in the venue density. Here we can observe that many data points gathered in less residence venue area. Assume several locations contain few residential venues, like commercial center or park, may differ from each other. These location might have diverse density. We could also see that in big city like New York, most of the retail stores located in area that have few residential venues. Popular spot for retail store location might be far from residential area.
After data cleaning and feature creation, we apply normalization to accelerate the training. Our ranking metric requires retail quality as a positive score, so we apply MinMaxScaler() to restrict the value of each feature within interval [0, 1].
Methodology and Metric
Due to our small data size, we apply cross validation during training. In each time of training, random samples 12% of data as testing set, and the rest is for training. Run over 1000 iterations, average the performance of each iteration to get the final result.
The metric to evaluate the performance is Normalized Discounted Cumulative Grain(NDCG@k), a famous metric frequently used in ranking problem. We first compute DCG@k (Discounted Cumulative Gain) for top-k highest ranked items, than normalize by Ideal rank DCG@k (IDCG@k).

NDCG@k metric measures the similarity between top-k highest rank item of prediction ranked list and the actual ranked list. If the top-k ranked item in prediction items list is close to the actual item list, then NDCG@k is close to 1; if the top-k ranked item in prediction items is strongly different from the actual item list, then NDCG@k is close to 0.
The more NDCG@k is close to 1, the better is the prediction ranking.
Individual feature performance
To analyze the elements that affects the retail store popularity, we care about the performance of ranking by each geographic feature. We assess the performance of each feature, and compare the behavior.

Table 1. Individual Feature Performance
To validate the efficiency, we compare the result with a random baseline which is created by randomly sort the testing items. The performances shows that total competitiveness outperform other features as the best geographic feature for retail store ranking. Density also give promising performance, while neighbor entropy and residence venue performs poorly. Here we apply polynomial linear regression for the assessment.
By the above result, it says that total competitiveness and density give better result than other feature. On the contrary, residence venue seems failed to encode the area popularity information for prediction, along with neighbor entropy. However, the combination of geographic features gives interesting result.
Combine geographic features to see if model improves
After combining multiple features and fed to the model, we found that the performance metric didn’t improve with more given features. This may caused from the noise contained in the features, to examine the combined features:
As we assumed, a retail store’s popularity could be correlated to consumer behavior, area popularity and number of competitors. In our geographic features, residential venues and density may contain similar information related to area popularity, and total competitiveness and competitiveness gives information about the number of competitors. We had found that when combining similar type features like total competitiveness and competitiveness, the performance would not improve but even drops.
Since we have no consumer behavior data here, to provide sufficient information for the prediction model at least one of the popularity data and competitiveness feature is required. We combine different pairs of these two type features and observe the performance as below table:

Table. 2 Combined geographic features performance
As opposed to our expectations, single total competitiveness performs best behavior. For the consideration of popularity feature, both residence venue and density gives promising result, while residence venue proves its effectiveness for providing popularity information.
Different model comparison
We apply four different models for comparison, and found that support vector regression performs the best. We set the parameter with RBF kernel, with C = 0.01.
For linear regression, we apply 20 degree polynomial features to get the best performance.
As for decision tree and support vector regression, no polynomial features is required, too much degree feature may even reduce the efficiency. Max depth is set to 30.
Neural network cost the most of time, we apply 5 degree polynomial features and 2 hidden layer with length 100 and 50.
Result
The combined features don’t show meaningful improvement comparing to individual features. This is not reasonable, since multiple features should provide more valuable information for predicting store popularity ranking.
We have applied complex models but the performance don’t improve further, this may due to coincidence on this specific data, or that our data size is too small which contains too much noise to provide enough information.
The overall performance has no significant improvement from past research, however we’ve proved that geographic features collected from Foursquare API is valuable to predict store popularity ranking on Google Map comments data.
Discussion
This project aimed to apply geographic data collected from a certain area, to predict the store’s popularity ranking within the place. The result shows that competitiveness and area popularity are actually correlated to retail store’s popularity.
The geographic data source of this project is limited to Foursquare API. Due to the quota of the API and the other Google Place API, the data size collected is limited to a range. To get better analysis for the retail store data, larger size of data and more features is required, however private data like revenues and total consumers during a time is hard to obtain. For a deeper exploration of retail store analysis, data collection may be the first challenge.
We are able to make decision for retail stores based on local geographic features within the city, however there are more field to be explored. In the experiment, model trained on an area would not perform well on another unseen area, even perform worse than random baseline. For business analyst, build an general model that could help decide the retail store location for an unexplored area is expected.
The next step for this project may be gather more relevant data, and analyze retail location ranking for different city, even area from different countries.
