# Breast-Cancer-Detection
Applying Different ML Classification models



## Libraries Used:  
  
1. Numpy  
2. Pandas  
3. Matplotlib  
4. Seaborn  
5. SciKit Learn   

## Data Set:  
  
You can Download the data set used in this project here at this link.  
"https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?select=breast-cancer.csv"  



## Flatform:  
  
For this project I used Google Colab. Which you can found here: https://colab.research.google.com/  



## About data:  

Data sonsists of 569 patients diagnosis results, test data. Out of these 212 patients are diagnosed with cancer. Data consists of 30 dependent variables and one dependent variable.  

We can see that there are strong correlations between some of these variables using Heatmap.  
![Heatmap-before-processing](https://github.com/balajiabcd/Breast-Cancer-Detection/blob/main/Heatmap-before-processing.png)  

Hence, we can remove these corrilated variables from our model.  
![Heatmap-after-processing](https://github.com/balajiabcd/Breast-Cancer-Detection/blob/main/Heatmap-after-processing.png)  

With the histplots between the variables we can see the there is clear distinction between data of patients who has cancer and data of patients who don't have cancer.  
![Histplot](https://github.com/balajiabcd/Breast-Cancer-Detection/blob/main/Histplot.png)  

With the scaterplot between the variables, we can see that it will be easy to separate the patients with cancer from those of without cancer.  
![Scattereplot](https://github.com/balajiabcd/Breast-Cancer-Detection/blob/main/Scatterplot.png)


## Note:  
Here to get the accuracy of classification model we used MAE(which is one of evaluation matrics of Regression model) as an indirect way of finding acuuracy of model.  




## Model Result:  


After trining and Deploying the different classification models and one regression model:  
Random Forest model with n_estimators = 400, criterion = "entropy", gave best results with 95.61% accuracy of detection.  
![Model](https://github.com/balajiabcd/Breast-Cancer-Detection/blob/main/randomforest%20model.png)








