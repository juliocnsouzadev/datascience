
# coding: utf-8

# In[2]:

#fat, short leg, barks
pig1 = [1,1,0]
pig2 = [1,1,0]
pig3 = [1,1,0]
dog1 = [1,1,1]
dog2 = [0,1,1]
dog3 = [0,1,1]

data = [pig1, pig2, pig3, dog1,dog2,dog3]

marks = ["pig","pig","pig","dog","dog","dog"]

incognito = [1,1,1]

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(data, marks)
print(model.predict(incognito))


# In[25]:

sueco1 = [200,0,1]
sueco2 = [198,0,1]
sueco3 = [196,1,1]
sueco4 = [194,0,1]
sueco5 = [192,0,1]
sueco6 = [190,0,1]
sueco7 = [188,1,1]
sueco8 = [186,0,1]
sueco9 = [184,0,1]
sueco10 = [182,0,1]
sueco11 = [180,0,1]
sueco12 = [178,0,1]
brasilieiro13 = [176,1,0]
brasilieiro14 = [174,1,0]
brasilieiro15 = [172,1,1]
brasilieiro16 = [170,0,1]
brasilieiro17 = [168,1,0]
brasilieiro18 = [166,1,0]
brasilieiro19 = [164,1,0]
brasilieiro20 = [162,1,1]
brasilieiro21 = [160,1,0]

pData = [sueco1,sueco2,sueco3,sueco4,sueco5,sueco6,sueco7,sueco8,sueco9,sueco10,sueco11,sueco12,brasilieiro13,brasilieiro14,brasilieiro15,brasilieiro16,brasilieiro17,brasilieiro18,brasilieiro19,brasilieiro20,brasilieiro21]
pMarks = ['sueco','sueco','sueco','sueco','sueco','sueco','sueco','sueco','sueco','sueco','sueco','sueco','brasilieiro','brasilieiro','brasilieiro','brasilieiro','brasilieiro','brasilieiro','brasilieiro','brasilieiro','brasilieiro']


c1 = [195,0,0]
c2 = [189,1,1]
c3 = [169,1,1]

testData = [c1,c2,c3]
testMarks = ['sueco', 'sueco', 'brasilerio']

from sklearn.naive_bayes import MultinomialNB
modelC = MultinomialNB()
modelC.fit(pData, pMarks)

result = modelC.predict(testData)

print(result)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



