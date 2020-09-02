##############################################################################
###################### Simple Linear Regression ##############################
##############################################################################



#Delivery_time -> Predict delivery time using sorting time 

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library

Delivery_Data=pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Simple_linear_Regression\\delivery_time.csv")
Delivery_Data.columns



################# EDA(explotary data analysis) ###########################


################ 1st moment business decision ######################

Delivery_Data.mean()#Delivery Time    16.790952
                   # Sorting Time      6.190476
Delivery_Data.median()#Delivery Time    17.83
                     #Sorting Time      6.00
Delivery_Data.mode()

################ 2nd moment busines decision ########################

Delivery_Data.var()  #Delivery Time    25.754619
                    #Salary             7.515510e+08                   
Delivery_Data.std() #YearsExperience        2.837888
                  #Salary             27414.429785                  

max(Delivery_Data['Delivery Time'])#29.0
max(Delivery_Data['Sorting Time'])#10
Range=max(Delivery_Data['Delivery Time'])-min(Delivery_Data['Delivery Time'])


################# 3rd and 4th moment business decision #################

Delivery_Data.skew()#Delivery Time    0.352390
                    #Sorting Time     0.047115  # both the data are positevely skewed

Delivery_Data.kurt()#Delivery Time    0.317960# implies thicker peak
                  #Sorting Time    -1.148455# since the kurtosis value is negative
                                              #implies both the distribution have wider peaks

#### Graphical representation   #########
                  
plt.hist(Delivery_Data['Sorting Time'])
plt.boxplot(Delivery_Data['Sorting Time'],0,"rs",0)


plt.hist(Delivery_Data['Delivery Time'])
plt.boxplot(Delivery_Data['Delivery Time'])

#Scatter plot
plt.plot(Delivery_Data['Sorting Time'],Delivery_Data['Delivery Time'],"bo");plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")


Delivery_Data['Delivery Time'].corr(Delivery_Data['Sorting Time']) # 0.8259972607955326 # correlation value between X and Y
# can be said as a moderate positive correlation


### or ### table format
Delivery_Data.corr()           #              Delivery Time  Sorting Time
#                          Delivery Time       1.000000      0.825997
 #                         Sorting Time        0.825997      1.000000

#or using numpy
np.corrcoef(Delivery_Data['Sorting Time'],Delivery_Data['Delivery Time'])
# gives the same result


import seaborn as sns
sns.pairplot(Delivery_Data)# gives the histogram and the scatter plot between the two variables




############## Model Preparing/ injecting the model #################



# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Delivery_Data['Delivery Time']~Delivery_Data['Sorting Time']",data=Delivery_Data).fit()

# For getting coefficients of the varibles used in equation
model.params#Intercept                        6.582734
             #Delivery_Data['Sorting Time']    1.649020
             
             
# P-values for the variables and R-squared value for prepared model
model.summary()#Adj. R-squared:                  0.666

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(Delivery_Data.iloc[:,1]) # Predicted values of Delivery Time using the model
 



# Visualization of regresion line over the scatter plot of YearsExperience and Salary
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt

#linear regression plot to c how the data is being destributed along the line
plt.scatter(x=Delivery_Data['Sorting Time'],y=Delivery_Data['Delivery Time'],color='red');plt.plot(Delivery_Data['Sorting Time'],pred,color='black');plt.xlabel('Sorting time');plt.ylabel('delivery time')






pred.corr(Delivery_Data['Delivery Time']) #0.8259972607955327
# Predicted vs actual values
plt.scatter(x=pred,y=Delivery_Data['Delivery Time'],);plt.xlabel("Predicted");plt.ylabel("Actual")




# Transforming variables for accuracy
model2 = smf.ols("Delivery_Data['Delivery Time']~np.log(Delivery_Data['Sorting Time'])",data=Delivery_Data).fit()
model2.params#Intercept                                1.159684
             #np.log(Delivery_Data['Sorting Time'])    9.043413
model2.summary()#Adj. R-squared:                  0.679

print(model2.conf_int(0.01)) # 99% confidence level

pred2 = model2.predict(pd.DataFrame(Delivery_Data['Sorting Time']))

pred2.corr(Delivery_Data['Delivery Time'])#0.8339325279256244
pred21 = model2.predict(Delivery_Data.iloc[:,1])
pred21
plt.scatter(x=Delivery_Data['Sorting Time'],y=Delivery_Data['Delivery Time'],color='green');plt.plot(Delivery_Data['Sorting Time'],pred21,color='blue');plt.xlabel('SortTime');plt.ylabel('DelTime')



#post transformation
# Exponential transformation
model3 = smf.ols("np.log(Delivery_Data['Delivery Time'])~Delivery_Data['Sorting Time']",data=Delivery_Data).fit()
model3.params
model3.summary()# Adj. R-squared:                  0.696,is increased as compared to log transformation
#R-squared:                       0.711
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(Delivery_Data['Sorting Time']))
pred_log
pred3=np.exp(pred_log)  
pred3
pred3.corr(Delivery_Data['Delivery Time'])#0.8085780108289262
plt.scatter(x=Delivery_Data['Sorting Time'],y=Delivery_Data['Delivery Time'],color='green');plt.plot(Delivery_Data['Sorting Time'],np.exp(pred_log),color='blue');plt.xlabel('Sorting Time');plt.ylabel('Delivery time')
resid_3 = pred3-Delivery_Data['Delivery Time']#error
resid_3




# getting residuals of the entire data set
Delivery_Data_resid = model.resid_pearson #error
Delivery_Data_resid 
plt.plot(model.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=Delivery_Data['Delivery Time']);plt.xlabel("Predicted");plt.ylabel("Actual")


#we can also check the other post  transformation 
# Quadratic model
#Salary_Data["YearsExperience_Sq"] = Salary_Data.YearsExperience*Salary_Data.YearsExperience
#Salary_Data#1 extra column will be formed
model_quad = smf.ols("Delivery_Data['Delivery Time']~Delivery_Data['Sorting Time']+Delivery_Data['Sorting Time']*Delivery_Data['Sorting Time']",data=Delivery_Data).fit()
model_quad.params#Intercept                        6.582734
                 #Delivery_Data['Sorting Time']    1.649020
model_quad.summary()#Adj. R-squared:                  0.666
pred_quad = model_quad.predict(Delivery_Data['Sorting Time'])

model_quad.conf_int(0.05) # 
plt.scatter(Delivery_Data['Sorting Time'],Delivery_Data['Delivery Time'],c="b");plt.plot(Delivery_Data['Sorting Time'],pred_quad,"r")

plt.scatter(np.arange(21),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 


## so we can c Expoential transformation,  is giving accuracy model having the highest R-squared value












############################### Implementing the Linear Regression model from sklearn library

from sklearn.linear_model import LinearRegression
import numpy as np
plt.scatter(Salary_Data.YearsExperience,Salary_Data.Salary)
model1 = LinearRegression()
model1.fit(Salary_Data.YearsExperience.values.reshape(-1,1),Salary_Data.Salary)
pred1 = model1.predict(Salary_Data.YearsExperience.values.reshape(-1,1))
# Adjusted R-Squared value
model1.score(Salary_Data.YearsExperience.values.reshape(-1,1),Salary_Data.Salary)# 0.9569566641435086
rmse1 = np.sqrt(np.mean((pred1-Salary_Data.Salary)**2)) # 32.760
model1.coef_
model1.intercept_

#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-wcat.AT),c="r")
plt.hlines(y=0,xmin=0,xmax=300) 
# checking normal distribution for residual
plt.hist(pred1-wcat.AT)

### Fitting Quadratic Regression 
wcat["Waist_sqrd"] = wcat.Waist*wcat.Waist
model2 = LinearRegression()
model2.fit(X = wcat.iloc[:,[0,2]],y=wcat.AT)
pred2 = model2.predict(wcat.iloc[:,[0,2]])
# Adjusted R-Squared value
model2.score(wcat.iloc[:,[0,2]],wcat.AT)# 0.67791
rmse2 = np.sqrt(np.mean((pred2-wcat.AT)**2)) # 32.366
model2.coef_
model2.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-wcat.AT),c="r")
plt.hlines(y=0,xmin=0,xmax=200)  
# Checking normal distribution
plt.hist(pred2-wcat.AT)
import pylab
import scipy.stats as st
st.probplot(pred2-wcat.AT,dist="norm",plot=pylab)

# Let us prepare a model by applying transformation on dependent variable
wcat["AT_sqrt"] = np.sqrt(wcat.AT)

model3 = LinearRegression()
model3.fit(X = wcat.iloc[:,[0,2]],y=wcat.AT_sqrt)
pred3 = model3.predict(wcat.iloc[:,[0,2]])
# Adjusted R-Squared value
model3.score(wcat.iloc[:,[0,2]],wcat.AT_sqrt)# 0.74051
rmse3 = np.sqrt(np.mean(((pred3)**2-wcat.AT)**2)) # 32.0507
model3.coef_
model3.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred3)**2,((pred3)**2-wcat.AT),c="r")
plt.hlines(y=0,xmin=0,xmax=300)  
# checking normal distribution for residuals 
plt.hist((pred3)**2-wcat.AT)
st.probplot((pred3)**2-wcat.AT,dist="norm",plot=pylab)

# Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
model4 = LinearRegression()
model4.fit(X = wcat.Waist.values.reshape(-1,1),y=wcat.AT_sqrt)
pred4 = model4.predict(wcat.Waist.values.reshape(-1,1))
# Adjusted R-Squared value
model4.score(wcat.Waist.values.reshape(-1,1),wcat.AT_sqrt)# 0.7096
rmse4 = np.sqrt(np.mean(((pred4)**2-wcat.AT)**2)) # 34.165
model4.coef_
model4.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred4)**2,((pred4)**2-wcat.AT),c="r")
plt.hlines(y=0,xmin=0,xmax=300)  

st.probplot((pred4)**2-wcat.AT,dist="norm",plot=pylab)

# Checking normal distribution for residuals 
plt.hist((pred4)**2-wcat.AT)

