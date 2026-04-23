
# coding: utf-8

# In[1]:

import pickle
import pandas as pd


# In[2]:


data=pd.read_csv(r"Mall_Customers (1).csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


print("Number of Rows",data.shape[0])
print("Number of columns",data.shape[1])


# In[7]:


data.info()


# In[8]:


data.isnull()


# In[9]:


data.isnull().sum()


# In[10]:


data.describe()


# In[11]:


data.columns


# In[21]:


X = data[['Annual Income (k$)',
       'Spending Score (1-100)']]


# In[25]:


from sklearn.cluster import KMeans


# In[26]:


k_means=KMeans()
k_means.fit(X)


# In[28]:


k_means=KMeans(n_clusters=5)
k_means.fit_predict(X)


# In[30]:


wcss=[]
for i in range(1,11):
    k_means = KMeans(n_clusters=i)
    k_means.fit(X)
    wcss.append(k_means.inertia_)


# In[31]:


wcss


# In[32]:


import matplotlib.pyplot as plt


# In[34]:


plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# In[35]:


X = data[['Annual Income (k$)',
       'Spending Score (1-100)']]


# In[37]:


k_means=KMeans(n_clusters=5,random_state=42)
y_means=k_means.fit_predict(X)


# In[38]:


y_means


# In[43]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='pink',label="Cluster 1")
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='orange',label="Cluster 2")
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='coral',label="Cluster 3")
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='blue',label="Cluster 4")
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='yellow',label="Cluster 5")
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100,c='red')
plt.title("Customer segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()


# In[45]:


k_means.predict([[15,39]])


# In[46]:


import joblib


# In[47]:


joblib.dump(k_means,"customer_segmentation.pkl")


# In[48]:


model=joblib.load("customer_segmentation.pkl")


# In[49]:


model.predict([[50,80]])

print("Model saved successfully!")
# # In[74]:


# from tkinter import *
# import joblib


# # In[80]:


# def show_entry_fields():
#     p1=int(e1.get())
#     p2=int(e2.get())
#     model = joblib.load('customer_segmentation')
#     result=model.predict([[p1,p2]])
#     print("This Customer belongs to cluster no: ",result[0])
#     if result[0]==0:
#         Label(master, text="Customers with medium annual income and medium annual spend"). grid(row=3,columnspan=2)
#     elif result[0]==1:
#         Label(master, text="Customers with high annual income and high annual spend"). grid(row=3,columnspan=2)
#     elif result[0]==2:
#         Label(master, text="Customers with low annual income and high annual spend"). grid(row=3,columnspan=2)
#     elif result[0]==3:
#         Label(master, text="Customers with high annual income but low annual spend"). grid(row=3,columnspan=2)
#     elif result[0]==4:
#         Label(master, text="Customers with low annual income and low annual spend"). grid(row=3,columnspan=2)

# master = Tk()
# master.title("Customer Segmentation Using Machine Learning")

# label=Label(master, text="Customer Segmentation Using Machine Learning", bg="pink",fg="white"). grid(row=0,columnspan=2)
# Label(master,text="Annual Income").grid(row=1)
# Label(master,text="Spending Score").grid(row=2)

# e1=Entry(master)
# e2=Entry(master)

# e1.grid(row=1, column=1)
# e2.grid(row=2, column=1)

# Button(master,text="Predict",command=show_entry_fields).grid(row=4,columnspan=2)

# master.mainloop()


# In[ ]:




