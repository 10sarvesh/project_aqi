#!/usr/bin/env python
# coding: utf-8

# The domain for this project is: "Environment and Climate Change".
# 
# Here, I have taken Air Quality Index Dataset where data taken is from last 5 years [2015 to 2020] in which various chemical constituents [PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene] in air are measured at different locations in India daily. Depending on the presence of this constituents in the air for a particular day, the air quality of the day can be categorized into 6 categories: Good, Satisfactory, Moderate, Poor, Very Poor, Severe 
# 
# The target variable here is "AQI_Bucket" which has these 6 categories [Good, Satisfactory, Moderate, Poor, Very Poor, Severe ] and is predicted by other feature variables in the dataset.
# 

# In[1]:


#importing all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # reading the data and getting the overall summary of data

# In[2]:


#reading and viewing the dataset; header=0 states the column labels are present in initial row

df_aqi=pd.read_csv("https://raw.githubusercontent.com/sarvesh0810/project_aqi/main/city_day.csv",header=0)
df_aqi


# In[3]:


df_aqi.head(10)          #displaying 1st 10 rows of dataframe


# In[4]:


df_aqi.tail()         #displays last 5 rows of dataframe


# In[5]:


df_aqi.shape        #gives (no. of observations, no. of features)


# In[6]:


df_aqi.columns      #reading all the columns in the given dataset


# In[7]:


df_aqi.dtypes       #gives the datatype of the resp. columns


# In[8]:


df_aqi.describe()      #gives the statistical summary of all columns


# In[9]:


pd.isnull(df_aqi).sum()        #gives the total no. of missing values in each column of dataframe


# # data cleaning part

# # dealing with missing values

# In[10]:


#dropping rows with missing values; axis=0: drop rows thresh: no. of non-NA values a row must have
#here thresh=10 meaning a row must have atleast 10 non-NA values

df_aqi.dropna(axis=0,thresh=10,inplace=True) 


# In[11]:


df_aqi.shape               #displays shape of new modified dataset


# In[12]:


pd.isnull(df_aqi).sum()      #again checking for missing values


# In[13]:


df_aqi.drop("Xylene",axis=1,inplace=True)     #dropping the column 'Xylene' as it has more than 50% missing values


# In[14]:


df_aqi.duplicated().sum()      #shows the number of duplicate rows present in the dataset


# In[15]:


#finding the means and std of all components and storing them in dataframe "mv"

data=[["PM2.5",round(df_aqi["PM2.5"].mean(),2),round(df_aqi["PM2.5"].std(),2)],
      ["PM10",round(df_aqi["PM10"].mean(),2),round(df_aqi["PM10"].std(),2)],
      ["NO",round(df_aqi["NO"].mean(),2),round(df_aqi["NO"].std(),2)],
      ["NO2",round(df_aqi["NO2"].mean(),2),round(df_aqi["NO2"].std(),2)],
      ["NOx",round(df_aqi["NOx"].mean(),2),round(df_aqi["NOx"].std(),2)],
      ["NH3",round(df_aqi["NH3"].mean(),2),round(df_aqi["NH3"].std(),2)],
      ["CO",round(df_aqi["CO"].mean(),2),round(df_aqi["CO"].std(),2)],
      ["SO2",round(df_aqi["SO2"].mean(),2),round(df_aqi["SO2"].std(),2)],
      ["O3",round(df_aqi["O3"].mean(),2),round(df_aqi["O3"].std(),2)],
      ["Benzene",round(df_aqi["Benzene"].mean(),2),round(df_aqi["Benzene"].std(),2)],
      ["Toluene",round(df_aqi["Toluene"].mean(),2),round(df_aqi["Toluene"].std(),2)]]

mv=pd.DataFrame(data,columns=["Element","Mean","Std"])
mv


# In[16]:


#Filling out the missing values in the columns with a random number from (mean-sd , mean+sd)

for i,data in df_aqi.iterrows():
    if pd.isnull(data["PM2.5"]):
        df_aqi.at[i,"PM2.5"]=np.random.uniform(df_aqi["PM2.5"].mean() - df_aqi["PM2.5"].std() , df_aqi["PM2.5"].mean() + df_aqi["PM2.5"].std())

for i,data in df_aqi.iterrows():
    if pd.isnull(data["PM10"]):
        df_aqi.at[i,"PM10"]=np.random.uniform(df_aqi["PM10"].mean() - df_aqi["PM10"].std() , df_aqi["PM10"].mean() + df_aqi["PM10"].std())

for i,data in df_aqi.iterrows():
    if pd.isnull(data["NO"]):
        df_aqi.at[i,"NO"]=np.random.uniform(df_aqi["NO"].mean() - df_aqi["NO"].std() , df_aqi["NO"].mean() + df_aqi["NO"].std())

for i,data in df_aqi.iterrows():
    if pd.isnull(data["NO2"]):
        df_aqi.at[i,"NO2"]=np.random.uniform(df_aqi["NO2"].mean() - df_aqi["NO2"].std() , df_aqi["NO2"].mean() + df_aqi["NO2"].std())

for i,data in df_aqi.iterrows():
    if pd.isnull(data["NOx"]):
        df_aqi.at[i,"NOx"]=np.random.uniform(df_aqi["NOx"].mean() - df_aqi["NOx"].std() , df_aqi["NOx"].mean() + df_aqi["NOx"].std())

for i,data in df_aqi.iterrows():
    if pd.isnull(data["NH3"]):
        df_aqi.at[i,"NH3"]=np.random.uniform(df_aqi["NH3"].mean() - df_aqi["NH3"].std() , df_aqi["NH3"].mean() + df_aqi["NH3"].std())
        
for i,data in df_aqi.iterrows():
    if pd.isnull(data["CO"]):
        df_aqi.at[i,"CO"]=np.random.uniform(df_aqi["CO"].mean() - df_aqi["CO"].std() , df_aqi["CO"].mean() + df_aqi["CO"].std())
        
for i,data in df_aqi.iterrows():
    if pd.isnull(data["SO2"]):
        df_aqi.at[i,"SO2"]=np.random.uniform(df_aqi["SO2"].mean() - df_aqi["SO2"].std() , df_aqi["SO2"].mean() + df_aqi["SO2"].std())
        
for i,data in df_aqi.iterrows():
    if pd.isnull(data["O3"]):
        df_aqi.at[i,"O3"]=np.random.uniform(df_aqi["O3"].mean() - df_aqi["O3"].std() , df_aqi["O3"].mean() + df_aqi["O3"].std())
        
for i,data in df_aqi.iterrows():
    if pd.isnull(data["Benzene"]):
        df_aqi.at[i,"Benzene"]=np.random.uniform(df_aqi["Benzene"].mean() - df_aqi["Benzene"].std() , df_aqi["Benzene"].mean() + df_aqi["Benzene"].std())
        
for i,data in df_aqi.iterrows():
    if pd.isnull(data["Toluene"]):
        df_aqi.at[i,"Toluene"]=np.random.uniform(df_aqi["Toluene"].mean() - df_aqi["Toluene"].std() , df_aqi["Toluene"].mean() + df_aqi["Toluene"].std())


# In[17]:


pd.isnull(df_aqi).sum()         #checking again for the missing values


# In[18]:


#dropping all those rows which have missing values for our target variable "AQI"

df_aqi.dropna(axis=0,inplace=True)


# In[19]:


#dealed all missing values and got a complete dataset wwith no missing values

pd.isnull(df_aqi).sum()


# # dealing with outliers

# In[20]:


def box(x):                            #defining a function to plot a boxplot for columns for finding outliers
    plt.figure(figsize=(7,5))
    plt.grid()
    sns.boxplot(x)
    
box(df_aqi["PM2.5"])                   #plotting boxplots for all columns one by one using function "box"
box(df_aqi["PM10"])
box(df_aqi["NO"])
box(df_aqi["NO2"])
box(df_aqi["NOx"])
box(df_aqi["NH3"])
box(df_aqi["CO"])
box(df_aqi["SO2"])
box(df_aqi["O3"])
box(df_aqi["Benzene"])
box(df_aqi["Toluene"])


# In[21]:


def qtl(x):                               #function for finding (0.05)th quantile and (0.95)th quantile
    lq=round(x.quantile(0.05),2)
    uq=round(x.quantile(0.95),2)
    print(lq,uq)
    
qtl(df_aqi["PM2.5"])                      #printing all quantile values one by one
qtl(df_aqi["PM10"])
qtl(df_aqi["NO"])
qtl(df_aqi["NO2"])
qtl(df_aqi["NOx"])
qtl(df_aqi["NH3"])
qtl(df_aqi["CO"])
qtl(df_aqi["SO2"])
qtl(df_aqi["O3"])
qtl(df_aqi["Benzene"])
qtl(df_aqi["Toluene"])


# In[22]:


#replacing the values lying outside (0.05,0.95)quantiles with 0.05th and 0.95th quantile

for i,data in df_aqi.iterrows():
    if data["PM2.5"]<df_aqi["PM2.5"].quantile(0.05):
        df_aqi.at[i,"PM2.5"]=df_aqi["PM2.5"].quantile(0.05)
    if data["PM2.5"]>df_aqi["PM2.5"].quantile(0.95):
        df_aqi.at[i,"PM2.5"]=df_aqi["PM2.5"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["PM10"]<df_aqi["PM10"].quantile(0.05):
        df_aqi.at[i,"PM10"]=df_aqi["PM10"].quantile(0.05)
    if data["PM10"]>df_aqi["PM10"].quantile(0.95):
        df_aqi.at[i,"PM10"]=df_aqi["PM10"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["NO"]<df_aqi["NO"].quantile(0.05):
        df_aqi.at[i,"NO"]=df_aqi["NO"].quantile(0.05)
    if data["NO"]>df_aqi["NO"].quantile(0.95):
        df_aqi.at[i,"NO"]=df_aqi["NO"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["NO2"]<df_aqi["NO2"].quantile(0.05):
        df_aqi.at[i,"NO2"]=df_aqi["NO2"].quantile(0.05)
    if data["NO2"]>df_aqi["NO2"].quantile(0.95):
        df_aqi.at[i,"NO2"]=df_aqi["NO2"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["NOx"]<df_aqi["NOx"].quantile(0.05):
        df_aqi.at[i,"NOx"]=df_aqi["NOx"].quantile(0.05)
    if data["NOx"]>df_aqi["NOx"].quantile(0.95):
        df_aqi.at[i,"NOx"]=df_aqi["NOx"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["NH3"]<df_aqi["NH3"].quantile(0.05):
        df_aqi.at[i,"NH3"]=df_aqi["NH3"].quantile(0.05)
    if data["NH3"]>df_aqi["NH3"].quantile(0.95):
        df_aqi.at[i,"NH3"]=df_aqi["NH3"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["CO"]<df_aqi["CO"].quantile(0.05):
        df_aqi.at[i,"CO"]=df_aqi["CO"].quantile(0.05)
    if data["CO"]>df_aqi["CO"].quantile(0.95):
        df_aqi.at[i,"CO"]=df_aqi["CO"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["SO2"]<df_aqi["SO2"].quantile(0.05):
        df_aqi.at[i,"SO2"]=df_aqi["SO2"].quantile(0.05)
    if data["SO2"]>df_aqi["SO2"].quantile(0.95):
        df_aqi.at[i,"SO2"]=df_aqi["SO2"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["O3"]<df_aqi["O3"].quantile(0.05):
        df_aqi.at[i,"O3"]=df_aqi["O3"].quantile(0.05)
    if data["O3"]>df_aqi["O3"].quantile(0.95):
        df_aqi.at[i,"O3"]=df_aqi["O3"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["Benzene"]<df_aqi["Benzene"].quantile(0.05):
        df_aqi.at[i,"Benzene"]=df_aqi["Benzene"].quantile(0.05)
    if data["Benzene"]>df_aqi["Benzene"].quantile(0.95):
        df_aqi.at[i,"Benzene"]=df_aqi["Benzene"].quantile(0.95)
        
for i,data in df_aqi.iterrows():
    if data["Toluene"]<df_aqi["Toluene"].quantile(0.05):
        df_aqi.at[i,"Toluene"]=df_aqi["Toluene"].quantile(0.05)
    if data["Toluene"]>df_aqi["Toluene"].quantile(0.95):
        df_aqi.at[i,"Toluene"]=df_aqi["Toluene"].quantile(0.95)
        


# In[23]:


df_aqi.sort_values(["Date","City"],ascending=True,inplace=True)      #sorting the values in ascending order by Date and City(alphabetical)


# # feature engineering: extracting more information from existing data for analysis

# In[24]:


for i,data in df_aqi.iterrows():     #i: specifies index position; data specifies data in df at corresponding index
    x=data['Date'].split("-")        #spliting date by '-' to get year and month
    df_aqi.at[i,"Year"]=x[2]         #assigning year to column 'year'
    df_aqi.at[i,"Month"]=x[1]        #assigning month to column 'month'


# In[25]:


df_aqi.reset_index(drop=True,inplace=True)        #reseting index and dropping original index


# In[26]:


df_aqi.dtypes


# In[27]:


summer=["03","04","05"]                        #creating seperate lists for different seasons based on months
monsoon=["06","07","08","09"]
autumn=["10","11"]
winter=["12","01","02"]

for i,data in df_aqi.iterrows():               #creating a new column 'Season'
    if data['Month'] in summer:
        df_aqi.at[i,"Season"]="Summer"
    if data['Month'] in monsoon:
        df_aqi.at[i,"Season"]="Monsoon"
    if data['Month'] in autumn:
        df_aqi.at[i,"Season"]="Autumn"
    if data['Month'] in winter:
        df_aqi.at[i,"Season"]="Winter"


# In[28]:


df_aqi


# In[29]:


df_aqi["City"].unique()    #displays the cities where AQI is measured


# In[30]:


#creating list of cities based on their geographical region

north_India=['Delhi', 'Lucknow','Amritsar']
west_coast=['Mumbai','Kochi','Thiruvananthapuram']
east_coast=['Chennai','Visakhapatnam','Kolkata']
central_India=['Bhopal','Ahmedabad','Jorapokhar']
north_east=['Guwahati','Shillong','Aizawl']
deccan_plateau=['Bengaluru','Hyderabad','Coimbatore']


# In[31]:


#creating a new column 'Geo. Region' to store Geographical region of cities based on their location

for i,data in df_aqi.iterrows():
    if data["City"] in north_India:
        df_aqi.at[i,"Geo. Region"]="North India"
    if data["City"] in west_coast:
        df_aqi.at[i,"Geo. Region"]="West Coast"
    if data["City"] in east_coast:
        df_aqi.at[i,"Geo. Region"]="East Coast"
    if data["City"] in central_India:
        df_aqi.at[i,"Geo. Region"]="Central India"
    if data["City"] in north_east:
        df_aqi.at[i,"Geo. Region"]="North East"
    if data["City"] in deccan_plateau:
        df_aqi.at [i,"Geo. Region"]="Deccan Plateau"


# In[32]:


df_aqi


# In[33]:


df_aqi.drop(["Date","Month"],axis=1,inplace=True)     #dropping columns that are not required


# In[34]:


#converting datatype of the columns having categorical data to 'category' datatype

df_aqi["AQI_Bucket"]=df_aqi["AQI_Bucket"].astype('category')
df_aqi["Year"]=df_aqi["Year"].astype('category')
df_aqi["Season"]=df_aqi["Season"].astype("category")
df_aqi["Geo. Region"]=df_aqi["Geo. Region"].astype("category")


# In[35]:


df_aqi.dtypes


# In[36]:


df_aqi["AQI_Bucket"]


# Some modification for storing categories-"AQI_Bucket" based on their magnitiudes

# In[37]:


#Here we 1st import CategoricalDtype from pandas.api.types and then store in type 'cat_bucket' in the order we want 
#And then in next step we change its type using function 'astype'

from pandas.api.types import CategoricalDtype
cat_bucket=CategoricalDtype(["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"],ordered=True)
df_aqi["AQI_Bucket"]=df_aqi["AQI_Bucket"].astype(cat_bucket)
df_aqi["AQI_Bucket"]


# # data analysis

# In[38]:


#displays total no. of days by "AQI Bucket" measured in Indian cities in last 5 years 

df_aqi["AQI_Bucket"].value_counts()


# In[39]:


#Bar plot for the above data
#X-axis has the catergories-"AQI Bucket" in decreasing order of their count
#Y-axis has the count for the respective catergories in last 5 years

plt.figure(figsize=(10,7))                                                    #figuresize
plt.xticks(fontsize=18)                                                       #increasing the fontsize of labels on x-axis
df_aqi["AQI_Bucket"].value_counts().plot(kind="bar",legend=True,grid=True)    #plotting bar graph for the counts of categories 


# In[40]:


#breakup of the above data
#shows yearwise breakup of the "AQI Bucket" and displays the count of respective categories in each year for all cities
#data is group yearly 1st and further by category of air in each year; .size() gives the size of data in respective group

df_aqi[["Year","AQI_Bucket"]].groupby(["Year","AQI_Bucket"]).size()   


# In[41]:


#plot for above data showing no. of days for each category in last 5 years

plt.xticks(fontsize=24)
df_aqi[["Year","AQI_Bucket"]].groupby(["AQI_Bucket","Year"]).size().plot(kind="bar",figsize=(30,18),grid=True)


# Conclusion from the above graph:
# 
# 1)Thus the above graph shows no. of "Good air quality days" are increasing from 2015 to 2020 which shows there
# are some measures taken by the government or other bodies to reduce air pollution.
# 
# 2)Also among all the categories "Satisfactory" and "Moderate" are in huge numbers indicating the overall air quality in 
# all cities is average and there is room for improvement.
# 
# 3)Also there is a drastic drop of "Severe air quality days" in year 2020 from 2019 which indicates the lockdown due to
# "COVID-19" pandemic had some impact on the air quality.
# 

# In[42]:


#shows the breakup of the "AQI Bucket" on the basis of different regions in India
#data is first groupby different regions and then overall count of each air category is displayed in each region

df_aqi[["Geo. Region","AQI_Bucket"]].groupby(["Geo. Region","AQI_Bucket"]).size()


# In[43]:


#plot for the above data
#barplot showing no. of days in each category in all regions with colors for each category
#colors chosen are standard AQI colors described by the Government 
#[Good:Green, Satisfactory:Yellow, Moderate:Orange, Poor:Red, Very Poor:Purple, Severe:Maroon]

plt.xticks(fontsize=26)
df_aqi[["Geo. Region","AQI_Bucket"]].groupby(["Geo. Region","AQI_Bucket"]).size().plot(kind="bar",figsize=(30,18),
                            grid=True,color=["green","yellow","orange","red","purple","maroon"])


# Conclusion:
# 
# 1)Thus from above plot we can see, both the coastal regions of India- West and East Coast have relatively less no. of days having poor to severe air quality. Also, the region in betweenn this two coasts-"Deccan Plateau" also has similar air quality.
# 
# 2)Whereas the regions- Central India and North India have more days in Poor to Severe air quality.
# 
# 3)The North East region has more no. of days in category Good to Moderate. This maybe due to the fact that the region is mostly rural and has fewer industries
# 
# So we can conclude that the Geographical Region of a paraticular area or city has some affect on its air quality.

# In[44]:


#Breakdown of the AQI_Bucket based on different seasons in India
#data grouped by season first and then the count is displayed for each air category in respective seasons

df_aqi[["AQI_Bucket","Season"]].groupby(["Season","AQI_Bucket"]).size()


# In[45]:


#plot for the above data
#plot shows no. of days in each air quality differentiated by different seasons

plt.xticks(fontsize=24)
df_aqi[["AQI_Bucket","Season"]].groupby(["Season","AQI_Bucket"]).size().plot(kind='bar',
                                        grid=True,figsize=(30,18),color=["green","yellow","orange","red","purple","maroon"])


# Conclusion:
# 
# 1)The plot shows that there are large no. of good to moderate days in season 'monsoon' followed by 'summer' season.
# 2)Whereas seasons-"Autumn" and "winter" both of which are post monsoon season have less no. of good air quality days and have large no. of Poor to severe air quality.
# 3)This maybe due to the fact that monsoon in India comes up with strong winds which pushes the bad air and the areas experience a good to moderate quality of air; whereas the climate in Winter and Autumn is mostly dry.
# 
# Thus, we can say that seasonal weathers of India have a certain impact on the air quality.

# In[46]:


df_aqi


# # beginning of machine learning process

# In[47]:


#Here we copy the data of "df_aqi" to "df_aqi_code" to keep our original data in one df and encoded data in other

df_aqi_code=df_aqi.copy()    


# In[48]:


df_aqi_code


# In[49]:


#converting categorical varaibles to datatype 'category'

df_aqi_code["City"]=df_aqi_code["City"].astype('category')
df_aqi_code["Year"]=df_aqi_code["Year"].astype('category')
df_aqi_code["Season"]=df_aqi_code["Season"].astype('category')
df_aqi_code["Geo. Region"]=df_aqi_code["Geo. Region"].astype('category')
df_aqi_code["AQI_Bucket"]=df_aqi_code["AQI_Bucket"].astype('category')

#converting all categories to codes
df_aqi_code["City_code"]=df_aqi_code["City"].cat.codes
df_aqi_code["Year_code"]=df_aqi_code["Year"].cat.codes
df_aqi_code["Season_code"]=df_aqi_code["Season"].cat.codes
df_aqi_code["Geo. Region_code"]=df_aqi_code["Geo. Region"].cat.codes
df_aqi_code["AQI_Bucket_code"]=df_aqi_code["AQI_Bucket"].cat.codes


# In[50]:


df_aqi_code


# In[51]:


#checking the columns of the new modified dataset

df_aqi_code.columns


# In[52]:


#dropping columns that are not required

columns_drop=['Geo. Region','City','Year','Season','AQI_Bucket']

df_aqi_code.drop(columns_drop,axis=1,inplace=True)  


# In[53]:


df_aqi_code.columns


# In[54]:


#saving the cleaned data to a new csv file-"aqi_cleaned"

#df_aqi_code.to_csv("D:\\BDA\\Sem 2\\Machine Learning 1\\Project\\city_day\\aqi_cleaned.csv",index=False)


# In[55]:


#reading the cleaned and encoded csv file

#df_aqi_code=pd.read_csv("D:\\BDA\\Sem 2\\Machine Learning 1\\Project\\city_day\\aqi_cleaned.csv",header=0)


# In[56]:


df_aqi_code


# In[57]:


#flattening all the values and storing them in a single array 

df_aqi_code.values.flatten()


# In[58]:


#finding number of negative elements in the entire dataset

sum(i<0 for i in df_aqi_code.values.flatten())


# In[59]:


#replacing all negative values with 0 as ML models don't accept negative values 

df_aqi_code[df_aqi_code<0]=0


# # feature selection

# In[60]:


#displaying the correlation matrix for the dataset

df_aqi_code.corr()


# In[61]:


#plotting the heatmap for the correlation matrix
#heatmap is a visualization plot of the correlation matrix which displays correlation between all the features 

hm=plt.cm.Greens                                                     #command for color required; here 'Greens' hence the heatmap has green shade
plt.figure(figsize=(18,11))                                          #figure size
plt.title("Heatmap for AQI data",fontsize=22)                        #title of heatmap
plt.xticks(fontsize=14)                                              #fontsize for x-axis
plt.yticks(fontsize=14)                                              #fontsize for y-axis
sns.heatmap(df_aqi_code.corr(),cmap=hm,linewidths=0.2,annot=True)    #plotting heatmap with seaborn sns library; line width: thin line between each block


# Conclusion:
# 
# From the above heatmap we can observe that no two features are very highly correlated.
# 
# 1) Variable AQI and AQI_Bucket have correlation value of 0.87 because AQI is the numerical value calculated and AQI_Bucket is the category or class of that numerical value.
# 2) Among all the chemical constituents found in the air, we can observe that "PM2.5" has the highest correlation value with AQI and AQI_Bucket which implies that "PM2.5" is the highest contributor to bad air in the environment.

# In[62]:


df_aqi_code.columns


# In[63]:


#storing feature variables in "X"

X=df_aqi_code.drop(["AQI","AQI_Bucket_code"],axis=1)
X


# In[64]:


#storing target variable in 'y'

y=df_aqi_code["AQI"]
y


# In[65]:


#dividing the data into train test data where size of test data is 33% of the entire dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=2)


# In[66]:


#using chi2 function to find top 10 most important features in the dataset which contribute to the target variable

from sklearn.feature_selection import chi2  #importing chi2 library
fscores=chi2(X_train,y_train)               #using the function chi2 for training data and storing it in 'fscores'
fscores                                     #1st list is the f-values and 2nd list is for p-values


# In[67]:


#here we use f-value for our analysis, hence we store data from 1st list in 'fvalue'

fvalue=pd.Series(fscores[0])                #convering the list into series 
fvalue.index=X_train.columns                #Here we set index as the column header for better understanding of data
fvalue.sort_values(ascending=False)         #sorting the data in descending order, since a feature with higher f-value is of more importance


# Here we select only top 10 features with higher f-values to feed into our model and drop the features -"Benzene", "City_code", "Year_code", "Geo. Region_code", "Season_code"            

# In[68]:


#dropping the insignificant columns from both train and test data 

X_train.drop(["Benzene", "City_code", "Year_code", "Geo. Region_code", "Season_code"],axis=1,inplace=True)
X_test.drop(["Benzene", "City_code", "Year_code", "Geo. Region_code", "Season_code"],axis=1,inplace=True)


# In[69]:


X_train


# In[70]:


X_test


# # applying MinMaxScaler to scale our data

# In[71]:


#applying MinMaxScaler to our dataset
#MinMaxScaler transforms all the data in the range of [0,1]

from sklearn.preprocessing import MinMaxScaler     #importing MinMaxScaler
ms=MinMaxScaler()                                  #storing it in an object 'ms'
X_train=ms.fit_transform(X_train)                  #fitting and transforming the train data
X_test=ms.transform(X_test)                        #transforming the test data


# In[72]:


print(X_train)
print("#########################################################")
print(X_test)


# # Applying LinearRegression

# In[73]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()                                
lr.fit(X_train,y_train)                             # fitting 
y_pred_lr=lr.predict(X_test)                        # prediction
print(y_pred_lr)


# In[74]:


from sklearn.metrics import mean_squared_error,r2_score
print("r2 score for linear regression=",round(r2_score(y_test,y_pred_lr),3))             # r square value
print("mean square error for linear regression=",mean_squared_error(y_test,y_pred_lr))   # mean square error


# # Applying Ordinary Least Squares (OLS) regression

# In[75]:


import statsmodels.api as sm

ols=sm.OLS(y_train,X_train).fit()
y_pred_ols=ols.predict(X_test)
print(y_pred_ols)


# In[76]:


print("R square value for OLS=",round(ols.rsquared,3))
print("Adjusted R square value for OLS=",round(ols.rsquared_adj,2))
print("Residual sum of square for OLS=",ols.ssr)
print("Mean square error for OLS",(ols.ssr/len(y)))             # mse=ssr/n  n: no. of observations


# # Applying Generalized Least Square (GLS) Regression

# In[77]:


gls=sm.GLS(y_train,X_train).fit()
y_pred_gls=gls.predict(X_test)
print(y_pred_gls)
print("R square value for GLS=",round(gls.rsquared,2))
print("Adjusted R square value for GLS=",round(gls.rsquared_adj,2))
print("Residual Sum of squares for GLS=",gls.ssr)
print("Mean square error for GLS=",(gls.ssr/len(y)))


# # Applying Support Vector Regression (SVR)

# In[78]:


from sklearn.svm import SVR
svr=SVR()
svr.fit(X_train,y_train)
y_pred_svr=svr.predict(X_test)
print(y_pred_svr)


# In[79]:


print("R square for SVR=",round(r2_score(y_test,y_pred_svr),2))
print("Mean square error for SVR=",mean_squared_error(y_test,y_pred_svr))


# # Applying RandomForestRegressor

# In[80]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr=rfr.predict(X_test)
print(y_pred_rfr)


# In[81]:


print("R square value for RandomForestRegressor=",round(r2_score(y_test,y_pred_rfr),2))
print("Mean square error for RandomForestRegressor=",mean_squared_error(y_test,y_pred_rfr))


# Thus here we can observe that:
# 
# 1) Among all regressors OLS,GLS and RandomForestRegressor perform the best with r-square value of 0.88.
# 2) RandomForestRegressor has the lowest value for mean_square_error hence we'll select it for further analysis.

# In[82]:


df_dep=pd.DataFrame(data=y_pred_rfr,columns=["AQI value"])       # storing the predicted values in dataframe df_dep

for i,data in df_dep.iterrows():
    if 0<=data["AQI value"]<=50:
        df_dep.at[i,"AQI category"]="Good"
    elif 51<=data["AQI value"]<=100:
        df_dep.at[i,"AQI category"]="Satisfactory"
    elif 101<=data["AQI value"]<=200:
        df_dep.at[i,"AQI category"]="Moderate"
    elif 201<=data["AQI value"]<=300:
        df_dep.at[i,"AQI category"]="Poor"
    elif 301<=data["AQI value"]<=400:
        df_dep.at[i,"AQI category"]="Very Poor"
    elif 401<=data["AQI value"]<=500:
        df_dep.at[i,"AQI category"]="Severe"
        
df_dep


# In[83]:


# deployment libraries

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
import streamlit as st


# In[84]:


app=FastAPI()    # creating an object of FASTAPI


# In[85]:


class request_body(BaseModel):              # class created to store datatype of variables given by user
    PM25:float                              # specifying datatype
    PM10:float
    NO:float
    NO2:float
    NOx:float
    NH3:float
    CO:float
    SO2:float
    O3:float
    Toluene:float


# In[86]:


# pickle library is used to dump(store) the learnt values

pickle.dump(rfr,open("model.pkl","wb"))       # dump the rfr model into variable model.pkl in "wb" (write binary) format so that it is unreadable


# In[87]:


# loading the dumped model

loaded_model=pickle.load(open("model.pkl","rb"))


# In[88]:


# creating user interface (UI)

def predict_input_page():
    st.title("Predicting Air Quality")           # title of the page
    
    PM25=st.text_input("PM2.5")                  # taking input from user through text input and storing them in variables
    PM10=st.text_input("PM10")
    NO=st.text_input("NO")
    NO2=st.text_input("NO2")
    NOx=st.text_input("NOx")
    NH3=st.text_input("NH3")
    CO=st.text_input("CO")
    SO2=st.text_input("SO2")
    O3=st.text_input("O3")
    Toluene=st.text_input("Toluene")
    
    ok=st.button("Predict AQI")                  # creating a button to predict
    
    try:                                         # checks if the suceeding condition holds true if not jumps to except
        if ok==True:                                 # if user presses the button
            test_data=np.array([[PM25,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Toluene]])
            class_idx=loaded_model.predict(test_data)[0]
            st.subheader(df_dep["AQI value"][class_idx])
            st.subheader(df_dep["AQI category"][class_idx])
    except:
        st.info("PLEASE ENTER APPROPRIATE DATA!!")


# In[89]:


# creating gate point for the website

@app.post("/predict")

def predict(data:request_body):
    test_data=[[
        data.PM25,
        data.PM10,
        data.NO,
        data.NO2,
        data.NOx,
        data.NH3,
        data.CO,
        data.SO2,
        data.O3,
        data.Toluene
    ]]
    
    class_idx=loaded_model.predict(test_data)[0]   # predicting the class
    return {"classname":df_dep["AQI value"][class_idx]}    # returning AQI value
    return {"classname":df_dep["AQI category"][class_idx]} # returning AQI category


# In[90]:


# input page

if __name__=="main":
    uvicorn.run(app,host="0.0.0.0",port=9000)


# In[ ]:




