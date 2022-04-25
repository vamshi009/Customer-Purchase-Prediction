import pandas
import pickle
import csv
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

f =  open("X_data_final.pkl", 'rb')
X_data = pickle.load(f)
f.close()

f =  open("Y_data_final.pkl", 'rb')
Y_data = pickle.load(f)
f.close()


X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)



#clf = tree.DecisionTreeClassifier()

clf = RandomForestClassifier(max_depth=10, random_state=42)
clf = clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

print(classification_report(Y_test, y_pred))

#r = export_text(clf)
#print(r)

f =  open("Evaldata.pkl", 'rb')
Eval_data = pickle.load(f)
f.close()

f = open("sessiondata.pkl", 'rb')
session_ids = pickle.load(f)
f.close()

Eval_pred = clf.predict(Eval_data)

print("len of Eval pred is ", len(Eval_pred))
print("len of session sis ", len(session_ids))

print(Eval_pred[-10:])
print(session_ids[-10:])


finalrows = []

temp = ["session_id","label"]
finalrows.append(temp)

Tcount = 0
Fcount = 0
for val in Eval_pred:
    if(val==1):
        Tcount = Tcount + 1
    if(val==0):
        Fcount = Fcount + 1

print("True Values in Eval are ", Tcount)
print("Fvalues in Eval are ", Fcount)

def write_to_file(rows, name):

    with open(name, 'w') as csvfile:
        writer = csv.writer(csvfile)

        for row in rows:
            writer.writerow(row)
        csvfile.close()
    print("Writeen to ", name)
    return


for i in range(len(Eval_pred)):
    temp = []
    temp.extend([session_ids[i], Eval_pred[i]])

    finalrows.append(temp)


write_to_file(finalrows, "Vamshi_Test_Submission_RandomForest.csv")
