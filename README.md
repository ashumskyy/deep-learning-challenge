# deep-learning-challenge


## Background

The nonprofit foundation wants a tool that can help it select the applicants for funding 
with the best chance of success in their ventures. Here we build a binary classifier that can predict 
whether applicants will be successful if funded.
From business team, we have received a CSV containing more than 34,000 organizations 
that have received funding over the years. 

Within this dataset are a number of columns that capture metadata about each organization, such as:
 - EIN and NAME — Identification columns
 - APPLICATION_TYPE — Alphabet Soup application type
 - AFFILIATION — Affiliated sector of industry
 - CLASSIFICATION — Government organization classification
 - USE_CASE — Use case for funding
 - ORGANIZATION — Organization type
 - STATUS — Active status
 - INCOME_AMT — Income classification
 - SPECIAL_CONSIDERATIONS — Special considerations for application
 - ASK_AMT - Funding amount requested
 - IS_SUCCESSFUL — Was the money used effectively
 
 ## Approach 
 
 ### Preprocessing the Data
 
 After reading the charity_data.csv file I identified the 'IS_SUCCESSFUL' variable as the target for the model.
 
 The 'EIN' and 'NAME' columns were dropped. 
 
 For columns with more than 10 unique values I used the number of data points for each unique value 
 to pick a cutoff point to bin "rare" categorical variables together in a new value, Other.
 
 Used pd.get_dummies() to encode categorical values.
 
 I split the preprocessed data into a features array, X, and a target array, y. 
 Then, used these arrays and the train_test_split function to split the data into training and testing datasets.
 
 At last, I scaled the training and testing features datasets by creating a StandardScaler instance, 
 fitting it to the training data, then using the transform function.
 
 ### Creating, Compiling, Training, Evaluating the Model
 
 I created the Neural Network by using TensorFlow and Keras. I used a Sequential model because it is appropriate 
 for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
 
 ### The model architecture
 	The goal to hit > 75% accuracy. 
 1. The model has input layer, two hidden layers and output layer. The first three layers have RELU activation, 
	the output layer has SIGMOID activation. For the units I used 32,32,16, 1 respectively.
	After training this model with epochs=50, the accuracy on the test data is 72.3%
 2. Afte dropping 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', adding the additional layer with units=24, activation='tanh', splitting the test and train data 70/30,
    and increasing the epochs=150 the result barely increased to 72.9%.
 3. After changing the activation from the new layer to 'swish', the accuracy increased to 73.1%.
 4. At last, by returning the NAME column in the original data, binning it that companies with value counts under 100 falls under the "Other" category, 
 	I was able to get accuracy result over 76% for training data, and 75.12% for testing data. 
