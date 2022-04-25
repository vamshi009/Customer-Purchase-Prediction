import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from collections import Counter



'''
In this program we perform exploratory data analysis
'''

df = pd.read_table("train.dat",  sep="," , header = 0)

print(df.head())

print(df.shape)

print("Unique item codes ", len(df['item_id_code'].unique()))

print("unique sessions ", len(df['session_id'].unique()))

print(df['label'].value_counts())

print("unique categories ", len(df['category'].unique()))

print("Slicing")

print(df[20:30])

y = df[20:30]
print(y['label'])

lbls = list(df['label'])
def get_pattern(list_labels):
    '''
    This function lets us know the buying pattern for actual purchases vs No purchases
    '''
    c = 0
    ans = []
    s  = ''
    count = 0
    for val in list_labels:

        if(c==1 and (s==val)):
            count = count + 1
        else:
            if(c==1):
                ans.append((s, count))
                count = 1
                s = val

        if(c==0):
            s = val
            c = 1
            count = 1

    return ans


#print(get_pattern(lbls))
pattern = get_pattern(lbls)
print(type(pattern[0][0]))

'''
Find th max and min number of points for successful purchase and no purchase 
'''

max = 0
min = 10000
for tup in pattern:
    if(str(tup[0])=='True'):
        x = tup[1]

        if(x>max):
            max = x
        if(x < min):
            min = x

print(max, min, " TRUE")
        

max = 0
min = 10000
for tup in pattern:
    if(str(tup[0])=='False'):
        x = tup[1]

        if(x>max):
            max = x
        if(x < min):
            min = x

print(max, min, " False")



'''
Finding number of NO  purchases having points below 200, 
'''
fcount = 0
fvalues = []
for val in pattern:
    if(str(val[0])=='False' and val[1]<=800):
        fcount = fcount + 1
        fvalues.append(val[1])

print("Number of flase data points ", fcount) 

Tcount = 0
Tvalues = []
for val in pattern:
    if(str(val[0])=='True' and val[1]<=200):
        Tcount = Tcount + 1
        Tvalues.append(val[1])

print("Number of True data points ", Tcount) 

'''
Number of flase data points  58848
Number of True data points  58848

This looks like a good balanced dataset for classification


'''
a = np.array(Tvalues)
#a = Tvalues
# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = [0, 100, 200, 300, 400, 500, 600, 700, 800])
 
# Show plot
#plt.show()
plt.savefig("TrueValues.png")


a = np.array(fvalues)
 
#a = fvalues
# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = [0, 100, 200, 300, 400, 500, 600, 700, 800])
 
# Show plot
#plt.show()
plt.savefig("FalseValues.png")

X_data = []
Y_data = []


def get_len_of_df(df):

    '''
    This feature to understand the number of timestamps user present in a session ID
    '''
    return df.shape[0]

def get_unique_category(df):

    '''
    To get the unique categories in a session
    '''

    return len(df['category'].unique())

def get_unique_items(df):
    '''
    No of unique items in a session
    '''

    return len(df['item_id_code'].unique())


def get_max_count_of_category(df):

    cats = list(df['category'])

    cdik = {}
    for val in cats:
        if(val in cdik):
            cdik[val] = cdik[val] + 1
        else:
            cdik[val] = 1

    vallist = []

    for val in cdik.values():
        vallist.append(val)

    #print("vallist is ", type(vallist), vallist)

    maxi = 0
    for val in vallist:
        if(val>maxi):
            maxi = val

    return maxi


def get_max_count_of_items(df):

    cats = list(df['item_id_code'])
    cdik = {}
    for val in cats:
        if(val in cdik):
            cdik[val] = cdik[val] + 1
        else:
            cdik[val] = 1

    vallist = []

    for val in cdik.values():
        vallist.append(val)

    maxi = 0
    for val in vallist:
        if(val>maxi):
            maxi = val

    return maxi

def get_seconds(string):
    slist = string.split(':')

    seconds = int(slist[0])*60*60 + int(slist[1])*60 + int(slist[2])
    return seconds


def get_time_diff(df):
    '''
    Calculates the number of seconds in a session
    '''

    times = list(df['timestamp'])

    s1 = times[0]
    s2 = times[-1]

    s1list = s1.split('T')
    s1Str = s1list[1][:-5]

    s2list = s2.split('T')
    s2Str = s2list[1][:-5]


    s1 = s1Str
    s2 = s2Str
    FMT = '%H:%M:%S'
    t1 = datetime.strptime(s1, FMT)
    t2 = datetime.strptime(s2, FMT)
    if(t1>t2):
        tdelta = t1 - t2
    else:
        tdelta = t2 - t1

    seconds = get_seconds(str(tdelta))

    return seconds

def get_avg_time(x_len, x_sec):
    '''
    Calculates average time spent per product
    '''
    return x_sec/x_len

def create_features(df, svalue):
    '''
    Generating the feature vectores here

    Class label is 1 for True values and 0 for False values 
    '''

    fvector = []

    x_len = get_len_of_df(df)
    fvector.append(x_len)

    x = get_unique_category(df)
    fvector.append(x)

    x = get_unique_items(df)
    fvector.append(x)

    x = get_max_count_of_category(df)
    fvector.append(x)

    x = get_max_count_of_items(df)
    fvector.append(x)

    x_sec = get_time_diff(df)
    fvector.append(x_sec)

    x_avg = get_avg_time(x_len, x_sec)
    fvector.append(x_avg)

    X_data.append(fvector)

    if(svalue==True):
        Y_data.append(1)
    else:
        Y_data.append(0)





def generate_data(list_labels, df):
    '''
    Getting session slices
    '''
    c = 0
    ans = []
    s  = ''
    count = 0
    Range = 0
    items = 0
    for val in list_labels:
        items = items + 1
        if(c==1 and (s==val)):
            count = count + 1
            if(items == len(list_labels)):
                ans.append((s, count))
                tempdf = df[Range: Range + count]
                create_features(tempdf, s)
                Range = Range + count
        else:
            if(c==1 or (items == len(list_labels))):
                ans.append((s, count))
                tempdf = df[Range: Range + count]
                create_features(tempdf, s)
                Range = Range + count
                count = 1
                s = val

        if(c==0):
            s = val
            c = 1
            count = 1

    return ans


generate_data(lbls, df)

print("len of X_data ", len(X_data))
print("len of Y_dtata ", len(Y_data))

'''
len of X_data  117697
len of Y_dtata  117697
'''

print(X_data[:100])
print(Y_data[:100])

with open("X_data_final.pkl", 'wb') as f:
    pickle.dump(X_data, f)
    f.close()


with open("Y_data_final.pkl", 'wb') as f:
    pickle.dump(Y_data, f)
    f.close()


dfeval = pd.read_table("test.dat",  sep="," , header = 0)

session_ids  = list(dfeval['session_id'])

Eval_data = []
session_names = []

def create_eval_features(df, svalue):
    '''

    Function which created features for final evaluation data and also stores the session ids
    '''
    fvector = []

    x_len = get_len_of_df(df)
    fvector.append(x_len)

    x = get_unique_category(df)
    fvector.append(x)

    x = get_unique_items(df)
    fvector.append(x)

    x = get_max_count_of_category(df)
    fvector.append(x)

    x = get_max_count_of_items(df)
    fvector.append(x)

    x_sec = get_time_diff(df)
    fvector.append(x_sec)

    x_avg = get_avg_time(x_len, x_sec)
    fvector.append(x_avg)


    Eval_data.append(fvector)
    session_names.append(svalue)



def generate_evaulation_data(list_labels):
    c = 0
    ans = []
    s  = ''
    count = 0
    Range = 0
    items = 0
    for val in list_labels:
        items = items + 1

        if(items == len(list_labels)):
            print("*"*77)
            print("val is ", val)
            print("s is ", s)
        if(c==1 and (s==val)):
            count = count + 1
            if(items == len(list_labels)):
                ans.append((s, count))
                tempdf = df[Range: Range + count]
                create_eval_features(tempdf, s)
                Range = Range + count
        else:
            if(c==1 or (items == len(list_labels))):
                ans.append((s, count))
                tempdf = df[Range: Range + count]
                create_eval_features(tempdf, s)
                Range = Range + count
                count = 1
                s = val

        if(c==0):
            s = val
            c = 1
            count = 1

    return ans


generate_evaulation_data(session_ids)
print("len of Eval dta is ", len(Eval_data))
print("len of sessions is ", len(session_names))

print(Eval_data[:10])
print(session_names[:10])

'''
len of Eval dta is  306825
len of sessions is  306825
'''

with open("Evaldata.pkl", 'wb') as f:
    pickle.dump(Eval_data, f)
    f.close()


with open("sessiondata.pkl", 'wb') as f:
    pickle.dump(session_names, f)
    f.close()
