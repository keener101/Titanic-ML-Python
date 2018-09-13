import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


train = pd.read_csv('titanic_train.csv')

plt.figure()
ax1 = plt.subplot(111)

ax1.set_title('Missing Data')

# look at null values in dataset

sns.heatmap(train.isnull(), cmap='viridis', yticklabels=False, cbar=False, ax=ax1)

# 'cabin' column has too many null values to be useful, so drop it.

train.drop(labels='Cabin', axis=1, inplace=True)

# drop 'Name', and 'Ticket' columns - likely useless

train.drop(labels=['Name', 'Ticket'], axis=1, inplace=True)


# now we have to decide what to do with age. Either impute based on mean, or by regression

# sns.heatmap(train.isnull(), cmap='viridis', yticklabels=False, cbar=False, ax=ax2)

#check if age can be predicted accurately.

#remove no-age columns, one of each will not be modified
noage = train[train['Age'].isnull()]
c_noage = train[train['Age'].isnull()]
ages = train[train['Age'] >= 0]
c_hasage = train[train['Age'] >= 0]

# plt.figure()
# ax3 = plt.subplot(111)
# sns.heatmap(ages.isnull(), cmap='viridis', yticklabels=False, cbar=False, ax=ax3)

#we also see that there is one passenger who is na for embarked - we will just drop that. 

ages.dropna(inplace=True)

#create dummy variables for qual vars

ages = pd.get_dummies(data=ages, columns=['Sex','Embarked', 'Pclass'])
noage = pd.get_dummies(data=noage, columns=['Sex','Embarked', 'Pclass'])


#set up linear regression categories - can't use 'Survived'

agesX = ages[['SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'
                , 'Pclass_1', 'Pclass_2', 'Pclass_3']]

agesy = ages['Age']

lm = LinearRegression()

lm.fit(agesX, agesy)

#look at data

print(lm.intercept_)

coeff_ages = pd.DataFrame(lm.coef_,agesX.columns,columns=['Coefficient'])
print(coeff_ages)

#examine p vals


X2 = sm.add_constant(agesX)
est = sm.OLS(agesy, X2)
est2 = est.fit()
print(est2.summary())

#2 predictors have p > 0.05 that aren't dummy vars. We will drop them. Parch and Fare

#rerun model

agesX = ages[['SibSp','Sex_female', 'Sex_male', 'Embarked_C','Embarked_Q', 'Embarked_S'
                , 'Pclass_1', 'Pclass_2', 'Pclass_3']]

agesy = ages['Age']

lm = LinearRegression()

lm.fit(agesX, agesy)

#look at data

print(lm.intercept_)

coeff_ages = pd.DataFrame(lm.coef_,agesX.columns,columns=['Coefficient'])
print(coeff_ages)

#examine p vals


X2 = sm.add_constant(agesX)
est = sm.OLS(agesy, X2)
est2 = est.fit()
print(est2.summary())



#clean up noage data to make predictions

noage = noage[['SibSp','Sex_female', 'Sex_male', 'Embarked_C','Embarked_Q', 'Embarked_S'
                , 'Pclass_1', 'Pclass_2', 'Pclass_3']]


#predict ages!

agepred = lm.predict(noage)

plt.figure()
ax5 = plt.subplot()

sns.distplot(agepred, ax=ax5)

ax5.set_title('Predicted Age (Training)')

#add predicted ages and reset data

c_noage['Age'] = agepred

cleaned_train = c_hasage.append(c_noage)

#drop the 2 straggler na values (like the missing embark found earlier )

cleaned_train.dropna(inplace=True)


#change ages < 0 to 0

plt.figure()
cleaned_ax = plt.subplot()

cleaned_train['Age'] = cleaned_train.Age.apply(lambda x: 0 if x < 0 else x)

sns.distplot(cleaned_train['Age'], ax=cleaned_ax)

cleaned_ax.set_title('Given + Imputed Ages (Training)')



#now we can model survival rate!


#create dummy variables on cleaned data

cleaned_train = pd.get_dummies(data=cleaned_train, columns=['Pclass', 'Sex', 'Embarked'])

#split data

cleanX = cleaned_train[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',
    'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    
cleany = cleaned_train['Survived']

logmodel = LogisticRegression()
logmodel.fit(cleanX,cleany)



print(logmodel.coef_)


logcoef= logmodel.coef_

logcoef = logcoef.ravel()


#print dataframe

logr = pd.DataFrame(logcoef, cleanX.columns, columns=['Coefficient'])
print(logr)








#----------------------------------










##Hooray! Now to make sure we can use our test data

test = pd.read_csv('titanic_test.csv')



## we need to drop cabin, and impute age

test.drop(labels='Cabin', axis=1, inplace=True)

#remove no-age columns, one of each will not be modified
tnoage = test[test['Age'].isnull()]
tc_noage = test[test['Age'].isnull()]
tages = test[test['Age'] >= 0]
tc_hasage = test[test['Age'] >= 0]


#create dummy variables for qual vars

tages = pd.get_dummies(data=tages, columns=['Sex','Embarked', 'Pclass'])
tnoage = pd.get_dummies(data=tnoage, columns=['Sex','Embarked', 'Pclass'])



#clean up noage data to make predictions

tnoage = tnoage[['SibSp','Sex_female', 'Sex_male', 'Embarked_C','Embarked_Q', 'Embarked_S'
                , 'Pclass_1', 'Pclass_2', 'Pclass_3']]


#predict ages!

tagepred = lm.predict(tnoage)

plt.figure()
ta_axes = plt.subplot()

sns.distplot(tagepred, ax=ta_axes)

ta_axes.set_title('Predicted Age (Test Values)')

#add predicted ages and reset data

tc_noage['Age'] = tagepred


cleaned_test = tc_hasage.append(tc_noage)



#152 lacks Fare data - is Pclass 3, so use mean.

p3only = cleaned_test[cleaned_test['Pclass'] == 3]
p3only = p3only.dropna()

print(p3only['Fare'].mean())


#P3 fare mean = `12.46

cleaned_test = cleaned_test.fillna(12.46)






# #change ages < 0 to 0

plt.figure()
cleaned_test_ax = plt.subplot()

cleaned_test['Age'] = cleaned_test.Age.apply(lambda x: 0 if x < 0 else x)

sns.distplot(cleaned_test['Age'], ax=cleaned_test_ax)

cleaned_test_ax.set_title('Given + Imputed Ages (Test)')

cleaned_test = pd.get_dummies(data=cleaned_test, columns=['Pclass', 'Sex', 'Embarked'])

# #split data

tcleanX = cleaned_test[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',
     'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    
    
# #predict!
    
finalpreds = logmodel.predict(tcleanX)

fpreds = pd.Series(finalpreds)

#make submission .csv for Kaggle

submission = cleaned_test['PassengerId'].to_frame()
submission.sort_values(by=['PassengerId'], inplace=True)

submission = submission.assign(Survived=fpreds)

submission.to_csv('keener-predict.csv', index=False)






























