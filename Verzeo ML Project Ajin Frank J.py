import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tkinter import *
from datetime import datetime
from tkinter import messagebox as Msg


SupWdwVar = Tk()
SupWdwVar.title("Tut Tul")
SupWdwVar.minsize(400, 300)
SupWdwVar.configure(background="white")

Label(SupWdwVar,
      text="Coronavirus Cases in INDIA Predictor",
      font="Cambria 32 bold",
      bg="white", fg="black",
      width=40, height=2).grid(row=1, column=0, columnspan=2)



Label(SupWdwVar,
      text="Now Enter the Date to predict: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=14, column=0, sticky=W)
Label(SupWdwVar,
      text="Thanks for the Information!!!!! ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=13, columnspan=2, sticky=W)

dat = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
dat.grid(row=14, column=1, sticky=W)

LblVar = Label(SupWdwVar, text="Go to the Link to get the following information :-", font="arial 16 bold")
LblVar.grid(row=4, column=0)

LblVa = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")

LblVa.grid(row=4, column=1)
LblVa.insert(0," https://www.worldometers.info/coronavirus")
#-------------------------------------------------------------------------------------
Label(SupWdwVar,
      text="Enter the Total Cases: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=5, column=0, sticky=W)

tc = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
tc.grid(row=5, column=1, sticky=W)

Label(SupWdwVar,
      text="Enter the Total Deaths: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=6, column=0, sticky=W)

td = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
td.grid(row=6, column=1, sticky=W)

Label(SupWdwVar,
      text="Enter the New Deaths Today: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=7, column=0, sticky=W)

nd = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
nd.grid(row=7, column=1, sticky=W)

Label(SupWdwVar,
      text="Enter the Total Cases per million Today: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=8, column=0, sticky=W)

tm = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
tm.grid(row=8, column=1, sticky=W)

Label(SupWdwVar,
      text="Enter the New Cases per million Today: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=9, column=0, sticky=W)

nm = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
nm.grid(row=9, column=1, sticky=W)

Label(SupWdwVar,
      text="Enter the Total Deaths per million Today: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=10, column=0, sticky=W)

tp = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
tp.grid(row=10, column=1, sticky=W)

Label(SupWdwVar,
      text="Enter the Total Tests done till Today: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=11, column=0, sticky=W)

tt = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
tt.grid(row=11, column=1, sticky=W)

Label(SupWdwVar,
      text="Enter the New Tests done Today: ",
      font="Cambria 16 bold",
      fg="black",
      anchor=E,
      width=35, height=1).grid(row=12, column=0, sticky=W)

nt = Entry(SupWdwVar,
                font="Times", width=70,
                bg="white", fg="grey")
nt.grid(row=12, column=1, sticky=W)


# 1.Import Dataset
data = pd.read_csv("owid-covid-data.csv")
# 2.Subset only INDIA column
realdata = data[data["location"] == "India"]

realdata_colums = ["total_cases", "new_cases", "total_deaths", "new_deaths",
"total_cases_per_million",
 "new_cases_per_million",
 "total_deaths_per_million", "new_deaths_per_million", "total_tests", "new_tests",
 "total_tests_per_thousand",
 "new_tests_per_thousand", "new_tests_smoothed",
"new_tests_smoothed_per_thousand",
 "stringency_index",
 "population", "population_density", "median_age", "aged_65_older", "aged_70_older",
"gdp_per_capita",
 "extreme_poverty",
 "cvd_death_rate", "diabetes_prevalence", "female_smokers", "male_smokers",
"handwashing_facilities",
 "hospital_beds_per_thousand", "life_expectancy"]
realdata = realdata.replace([0, ' ', 'NULL'], np.nan)
realdata.dropna(thresh=realdata.shape[0] * 0.5, how='all', axis=1)
realdata = realdata.replace([0, ' ', 'NULL'], np.nan)
for i in realdata_colums:
 colmean = realdata[i].mean()
 realdata[i].fillna(colmean, inplace=True)
realcol_colums = ["iso_code", "continent", "location", "tests_units"]
for j in realcol_colums:
 catcolmode = realdata[j].mode()[0]
 realdata[j].fillna(catcolmode, inplace=True)
null_columns = realdata.columns[realdata.isnull().any()]
print(realdata[null_columns].isna().sum())

# 6.Datetime Conversion
import datetime as dt
realdata["date"] = pd.to_datetime(realdata["date"])
realdata["date"] = realdata["date"].map(dt.datetime.toordinal)
#7. Dropping Categorical Columns
realdata.drop(["iso_code", "continent", "location", "tests_units"], axis=1, inplace=True)

def getdate():
    date = dat.get()
    totalcases=tc.get()
    totaldeath=td.get()
    newdeaths=nd.get()
    totalcasespermillion=tm.get()
    newcasespermillion=nm.get()
    totaldeathspermillion=tp.get()
    totaltests=tt.get()
    newtests=nt.get()
    totalcases1=float(totalcases)
    totaldeath1 = float(totaldeath)
    newdeaths1 = float(newdeaths)
    totalcasespermillion1 = float(totalcasespermillion)
    newcasespermillion1 = float(newcasespermillion)
    totaldeathspermillion1 = float(totaldeathspermillion)
    totaltests1=float(totaltests)
    newtests1=float(newtests)


    print(date)
    d = datetime.strptime(date, '%Y-%m-%d')
    finaldate = d.toordinal()
    print(finaldate)
    # 8.Model Building
    y = realdata['new_cases'].values
    x = realdata.drop(['new_cases'], axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
    modelsel = LinearRegression()
    modelsel.fit(x_train, y_train)
    newdata = pd.DataFrame({"date": finaldate, "total_cases": [totalcases1], "total_deaths": totaldeath1, "new_deaths": newdeaths1,
                            "total_cases_per_million": totalcasespermillion1,
                            "new_cases_per_million": newcasespermillion1,
                            "total_deaths_per_million": totaldeathspermillion1, "new_deaths_per_million": 3, "total_tests": totaltests1,
                            "new_tests": newtests1,
                            "total_tests_per_thousand": 216.377,
                            "new_tests_per_thousand": 1.324, "new_tests_smoothed": 1744230,
                            "new_tests_smoothed_per_thousand": 1.264,
                            "stringency_index": 73.61,
                            "population": 1380004385, "population_density": 450.419, "median_age": 28.2,
                            "aged_65_older": 5.989, "aged_70_older": 3.414, "gdp_per_capita": 6426.674,
                            "extreme_poverty": 21.2,
                            "cvd_death_rate": 282.28, "diabetes_prevalence": 10.39, "female_smokers": 1.9,
                            "male_smokers": 20.6, "handwashing_facilities": 59.55,
                            "hospital_beds_per_thousand": 0.53, "life_expectancy": 69.66})
    y_pred = modelsel.predict(newdata)
    print("Predicted No of Cases:", y_pred)
    print("Accuracy of LinearRegression Model:", modelsel.score(x_test, y_test))

    Y = realdata['new_cases'].values
    X = realdata.drop(['new_cases'], axis=1).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=45)
    model2sel = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=45)
    model2sel.fit(X_train, Y_train)
    y_2pred = model2sel.predict(X_test)
    print("Accuracy of RandomForestRegressor Model:", model2sel.score(x_test, y_test))
    newdata2 = pd.DataFrame({"date": finaldate, "total_cases": [21892676], "total_deaths": 238265, "new_deaths": 4194,
                             "total_cases_per_million": 15729,
                             "new_cases_per_million": 288,
                             "total_deaths_per_million": 171, "new_deaths_per_million": 3, "total_tests": 298601699,
                             "new_tests": 1826490,
                             "total_tests_per_thousand": 216.377,
                             "new_tests_per_thousand": 1.324, "new_tests_smoothed": 1744230,
                             "new_tests_smoothed_per_thousand": 1.264,
                             "stringency_index": 73.61,
                             "population": 1380004385, "population_density": 450.419, "median_age": 28.2,
                             "aged_65_older": 5.989, "aged_70_older": 3.414, "gdp_per_capita": 6426.674,
                             "extreme_poverty": 21.2,
                             "cvd_death_rate": 282.28, "diabetes_prevalence": 10.39, "female_smokers": 1.9,
                             "male_smokers": 20.6, "handwashing_facilities": 59.55,
                             "hospital_beds_per_thousand": 0.53, "life_expectancy": 69.66})
    y_2pred = modelsel.predict(newdata2)
    print("Predicted No of Cases:", y_2pred)
    Msg.showinfo("Predicted No of Cases",y_pred)

print("------------Start------------")
Button(SupWdwVar, text="Submit",
    bg="white", fg="black",
    width=10, command=getdate,
    font="arial 40 bold").grid(row=16, columnspan=2)

SupWdwVar.mainloop()



