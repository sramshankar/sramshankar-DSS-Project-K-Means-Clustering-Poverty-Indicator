import numpy as np
from datascience import *
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
sns.set()
from sklearn.cluster import KMeans
dataframe1 = pd.read_csv('train_values_wJZrCmI-20211006-035621.csv')
table = Table.read_table('train_values_wJZrCmI-20211006-035621.csv') #imports

df = dataframe1.drop(columns = ["age","female","married","religion","relationship_to_hh_head", "employment_category_last_year","employment_type_last_year","share_hh_income_provided", "income_ag_livestock_last_year","income_friends_family_last_year","income_government_last_year","income_own_business_last_year","income_private_sector_last_year","income_public_sector_last_year", "num_times_borrowed_last_year","borrowing_recency","informal_savings","cash_property_savings","bank_interest_rate","mm_interest_rate","mfi_interest_rate","other_fsp_interest_rate","num_shocks_last_year","avg_shock_strength_last_year","borrowed_for_daily_expenses_last_year","borrowed_for_home_or_biz_last_year","phone_technology","can_call","can_text","can_use_internet","can_make_transaction","phone_ownership","advanced_phone_use","reg_bank_acct","reg_mm_acct","reg_formal_nbfi_account","financially_included","active_bank_user","active_mm_user","active_formal_nbfi_user","active_informal_nbfi_user","nonreg_active_mm_user","num_formal_institutions_last_year","num_informal_institutions_last_year","num_financial_activities_last_year"])
df = df.drop_duplicates().dropna()

df['is_urban'] = df['is_urban'].astype(int)
df['education_level'] =df['education_level'].astype(int)
df['literacy'] =df['literacy'].astype(int)
for (columnName, columnData) in df.iteritems():
    if columnName != 'country':
        df[columnName] =df[columnName].astype(int)   
        
df.info() #shows that data cleaning was successful
#everything is numeric except country

dfkmeans = KMeans(n_clusters = 4, random_state= 0)
x1 = df[["is_urban","education_level","literacy"]]
y1 = dfkmeans.fit_predict(x1)

x1["cluster"] = y1
x1.head()

#new array for each point's color depending on cluster
clusterarray = np.array(x1["cluster"])
colorsarray = []
for i in np.arange(len(clusterarray)):
    if clusterarray[i -1] ==0:
        colorsarray.append("green")
    elif clusterarray[i -1] ==1:
        colorsarray.append("blue")
    elif clusterarray[i-1] ==2:
        colorsarray.append("red")
    elif clusterarray[i-1] ==3:
        colorsarray.append("yellow")

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
 
# Creating dataset
z = np.array(x1["is_urban"])
x = np.array(x1["education_level"])
y = np.array(x1["literacy"])
 
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = colorsarray)# color = x1['cluster'])
#plt.title("simple 3D scatter plot")
 
# show plot
plt.show()

x1_grouped = x1.groupby(["cluster"], as_index=False).mean()
x1_grouped.head()

x1_grouped.plot.bar(x="cluster", y = "is_urban")

x1_grouped.plot.bar(x="cluster", y = "education_level")

x1_grouped.plot.bar(x="cluster", y = "literacy")

color1 = np.array(x1["cluster"])
groups = Table().with_columns("cluster",color1, "is_urban", z, "education_level",x, "literacy", y)
clustergroup = groups.group("cluster")
clustergroup.show()
