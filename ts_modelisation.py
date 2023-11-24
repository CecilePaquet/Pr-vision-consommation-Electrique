import pipes
import streamlit as st

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#joblib.dump(clf, "model")
from PIL import Image

#%matplotlib inline # pour importance des données # sinon importer directement la copie d'écran
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import joblib
import statsmodels.api as sm
import pickle


from prophet import Prophet


dfMW_groupby_date = pd.read_csv("dfMW_groupby_date.csv", sep = ",")
dfMW_nationale_date_somme = pd.read_csv("dfMW_nationale_date_somme.csv", sep = ",")
dfMJ = pd.read_csv("dfMJ.csv", sep = ",")
dfMW_temp_nationale_date_somme=pd.read_csv('dfMW_temp_nationale_date_somme.csv', sep=',')
réel_14 = pd.read_csv("réel_14.csv", index_col = 0)
ts_21_22_deb_log = pd.read_csv("ts_21_22_deb_log.csv", index_col = 0)
ts_temp_deb = pd.read_csv("ts_temp_deb.csv", index_col = 0)
ts_temp_fin = pd.read_csv("ts_temp_fin.csv", index_col = 0)
df_métriques_sans_reclim = pd.read_csv("df_métriques_sans_reclim.csv", index_col=0)
df_métriques_reclim=pd.read_csv("df_métriques_reclim.csv", index_col=0)
pred_tb_h_1_14_suite = pd.read_csv("pred_tb_h_1_14_suite.csv", index_col=0)
réel_h_14_suite = pd.read_csv("réel_h_14_suite.csv", index_col=0)
pred_pro_h_8_1 = pd.read_csv("pred_pro_h_8_1.csv", index_col=0)
réel_h_1 = pd.read_csv("réel_h_1.csv", index_col=0)
pred_pro_8_1 = pd.read_csv("pred_pro_8_1.csv", index_col=0)
réel_1 = pd.read_csv("réel_1.csv", index_col=0)
df_compar_modèles = pd.read_csv("df_compar_modèles.csv", index_col=0)
df_metrics_machine_learning = pd.read_csv("df_metrics_machine_learning.csv", index_col=0)



# A importer dans autre .py :
# m_pro_h_loaded
# m_pro_j_loaded
# svr_loaded
# réel_8
from prophet import Prophet

réel_h_8 = pd.read_csv("réel_h_8.csv", index_col=0)
réel_h_8["ds"] = réel_h_8.index
réel_h_8["y"]= réel_h_8["MW"]

réel_8 = pd.read_csv("réel_8.csv", index_col=0)

réel_8["ds"] = réel_8.index
réel_8["y"]= réel_8["Consommation (MW)"]

st.write(réel_8)












######################
###Plan de l'appli####
######################
st.sidebar.subheader("Le réseau électrique français, data analyse de la consommation")
page1 = "Time series"



pages = [page1]
select_page = st.sidebar.radio("", pages)

st.sidebar.info(
"Auteurs : "
"Cécile Paquet "
"[linkedIn](https://linkedin.com/in/cécile-paquet-198816144/), "
"Alexis Mouradoff, "
"Abdessamad Moussaoui "

"\n\n"
"Formation continue Data Analyst Avril 2023, "
"[DataScientest](https://datascientest.com/)"

"\n\n"
"Données :"
"[RTE](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure), "
"[Weathnews.fr]((https://odre.opendatasoft.com/explore/dataset/temperature-quotidienne-departementale/table/?disjunctive.departement&sort=date_obs&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJhcmVhcmFuZ2UiLCJmdW5jIjoiQ09VTlQiLCJ5QXhpcyI6InRtb3kiLCJzY2llbnRpZmljRGlz))"
)

##########################
####Projet et résultats####
##########################



if select_page == page1 : 
  st.write("# Prévision de la consommation")
  #st.markdown("<h3 style='text-align: center; color: grey;'>Prévision de la consommation électrique française <br>  </h3>", unsafe_allow_html=True)
  st.write("Type de machine learning") 

    
  if st.checkbox('Modèles de séries temporelles'):


     st.write("#### Meilleur résultat par étendue de prédiction et par pas")
     ques_machine=st.radio("Choix de l'étendue / Choix du pas", ('14 jours / Pas journalier', '14 jours / Pas horaire','1 an / Pas journalier','1 an / Pas horaire')) 
     if ques_machine =='14 jours / Pas journalier':  
       if st.checkbox("Entraînement du modèle et affichage des métriques pour SARIMAX_temp_1_14") :
             model = sm.tsa.statespace.SARIMAX(ts_21_22_deb_log,
                                      exog = ts_temp_deb,
                                      order=(0, 1, 0),
                                      seasonal_order=(1, 0, 1, 7),
                                      time_varying_regression = True,
                                      mle_regression = False,
                                      enforce_stationarity=True,
                                      enforce_invertibility=True)
             results_sarimax_plus=model.fit()
             pred_sarx_1_14=np.exp(results_sarimax_plus.predict(start="2022-02-15", end="2022-02-28", exog=ts_temp_fin, dynamic=False))
             pred_sarx_1_14.index = réel_14.index

           
             mae_sarx_1_14 = mean_absolute_error(réel_14, pred_sarx_1_14)
             rmse_sarx_1_14 = mean_squared_error(réel_14, pred_sarx_1_14,squared=False)
             MWR = [elt for elt in réel_14["Consommation (MW)"]]
             MWP = [elt for elt in pred_sarx_1_14]
             result_sarx_1_14 = pearsonr(MWR, MWP)
             pv_sarx_1_14 = result_sarx_1_14[1]
             st.write("P-value_SARIMAX_1_14 : ", pv_sarx_1_14)
             st.write("RMSE_SARIMAX_1_14 :", rmse_sarx_1_14)
             st.write("MAE_SARIMAX_1_14 :", mae_sarx_1_14)

       if st.checkbox("Afficher la courbe et le scatter prédiction versus réalité pour SARIMAX_temp_1_14"):
             
             fig = plt.figure(figsize = (14, 7))
             #plt.figure(figsize = (14, 7))
             plt.plot(réel_14, color = 'black', label = "Réalité") # les 14 jours de vraie conso
             plt.plot(pred_sarx_1_14, color='orange', label = "Prédiction SARIMAX") # la prédiction SARIMAx de conso sur ces 14 jours
             plt.title('Prédiction journalière SARIMAX de consommation sur 14 jours versus Réalité')
             plt.ylabel("Consommation en MW")
             plt.xticks(np.arange(0, 14), ["15-02-22", "16-02-22", "17-02-22", "18-02-22", "19-02-22", "20-02-22", "21-02-22", "22-02-22", "23-02-22", "24-02-22", "25-02-22", "26-02-22", "27-02-22", "28-02-22"])
             plt.legend()
             st.pyplot(fig)
             st.write("La courbe de prédiction de SARIMAX intégrant la température en facteur exogène suit remarquablement bien celle des consommations réelles, tout en les surestimant légèrement.")

             fig = plt.figure(figsize = (14, 7))
             plt.scatter(réel_14, pred_sarx_1_14, color = "orange", label = "Prédiction SARIMAX_1_14")
             plt.xlabel("Réalité")
             plt.ylabel("Prédictions")
             plt.plot([55000, 70000], [55000, 70000], color = 'black')
             plt.title("Consommation en MW : Prédictions SARIMAX_température VS Réalité")
             plt.legend()
             st.pyplot(fig)    

     if ques_machine =='14 jours / Pas horaire':
        if st.checkbox("Entraînement du modèle et affichage des métriques pour TBATS_h_1_14_suite") :
          mae_tb_h_1_14_suite = mean_absolute_error(réel_h_14_suite, pred_tb_h_1_14_suite)
          rmse_tb_h_1_14_suite = mean_squared_error(réel_h_14_suite, pred_tb_h_1_14_suite,squared=False)
          MWR = [elt for elt in réel_h_14_suite["MW"]]
          MWP = [elt for elt in pred_tb_h_1_14_suite["MW"]]
          result_tb_h_1_14_suite = pearsonr(MWR, MWP)
          pv_tb_h_1_14_suite = result_tb_h_1_14_suite[1]
          st.write("P-value_TBATS_h_1_14_suite : ", pv_tb_h_1_14_suite)
          st.write("RMSE_TBATS_h_1_14_suite :", rmse_tb_h_1_14_suite)
          st.write("MAE_TBATS_h_1_14_suite :", mae_tb_h_1_14_suite)     

             

        if st.checkbox("Afficher la courbe et le scatter prédiction versus réalité pour TBATS_h_1_14_suite"):
             fig = plt.figure(figsize = (14, 7))
             #plt.figure(figsize = (14, 7))            
             plt.plot(réel_h_14_suite, color = 'black', label = "Réalité") 
             plt.plot(pred_tb_h_1_14_suite, color='orange', label = "Prédiction TBATS_h_1_14_suite") 
             plt.title('Entraînement horaire sur 2021-2022 et prédiction horaire TBATS_h_1_14_suite de consommation sur 14 jours versus Réalité')
             plt.ylabel("Consommation en MW")
             plt.xticks(np.arange(0, 336, 24), ["01-03-22", "02-03-22", "03-03-22", "04-03-22", "05-03-22", "06-03-22", "07-03-22", "08-03-22", "09-03-22", "10-03-22", "11-03-22", "12-03-22", "13-03-22", "14-03-22"])
             plt.legend()
             plt.show();
             st.pyplot(fig)

             fig = plt.figure(figsize = (14, 7))
             plt.scatter(réel_h_14_suite, pred_tb_h_1_14_suite, s = 7, color = "orange", label = "Prédiction TBATS_h_1_14_suite")
             plt.xlabel("Réalité")
             plt.ylabel("Prédictions")
             plt.plot([50000, 75000], [50000, 75000], color = 'black')
             plt.title("Consommation en MW : Prédictions de TBATS_h_1_14_suite VS Réalité")
             plt.legend();
             st.pyplot(fig)
        

     if ques_machine =='1 an / Pas journalier':
        if st.checkbox("Entraînement du modèle et affichage des métriques pour PROPHET_8_1") :
             
             mae_pro_8_1 = mean_absolute_error(réel_1, pred_pro_8_1["yhat"])
             rmse_pro_8_1 = mean_squared_error(réel_1, pred_pro_8_1["yhat"], squared=False)
             st.write("RMSE_PROPHET_8_1 :", rmse_pro_8_1)
             st.write("MAE_PROPHET_8_1 :", mae_pro_8_1)
             result_pro_8_1 = pearsonr(réel_1["Consommation (MW)"], pred_pro_8_1["yhat"])
             pv_pro_8_1 = result_pro_8_1[1]
             st.write("P-value PROPHET_8_1 : ", pv_pro_8_1)

        if st.checkbox("Afficher la courbe et le scatter prédiction versus réalité pour PROPHET_8_1"):
             fig = plt.figure(figsize = (14, 7))
             plt.plot(réel_1, color = 'black', label = "Réalité") # l'année' de vraie conso
             plt.plot(pred_pro_8_1, color='orange', linestyle = "--", label = "Prédiction PROPHET") # la prédiction PROPHET de conso sur l'année
             plt.title('Entraînement sur 2013-2021 et prédiction journalière PROPHET de consommation sur 2021-2022 versus Réalité')
             plt.ylabel("Consommation en MW")
             plt.xticks(np.arange(0, 364, 60), ["mars 2021", "mai 2021", "juillet 2021", "septembre 2021", "novembre 2021", "janvier 2022", "mars 2022"])
             plt.legend()
             plt.show()
             st.pyplot(fig)

             fig = plt.figure(figsize = (14, 7))
             plt.scatter(réel_1, pred_pro_8_1, color = "orange", label = "Prédiction PROPHET")
             plt.xlabel("Réalité")
             plt.ylabel("Prédictions")
             plt.plot([25000, 80000], [25000, 80000], color = 'black')
             plt.title("Consommation en MW : Prédictions de PROPHET_j_8_1 VS Réalité")
             plt.legend()
             st.pyplot(fig)
        

     if ques_machine =='1 an / Pas horaire':
        if st.checkbox("Entraînement du modèle et affichage des métriques pour PROPHET_h_8_1") :
             mae_pro_h_8_1 = mean_absolute_error(réel_h_1, pred_pro_h_8_1)
             rmse_pro_h_8_1 = mean_squared_error(réel_h_1, pred_pro_h_8_1, squared=False)
             st.write("RMSE_prophet_h_8_1 :", rmse_pro_h_8_1)
             st.write("MAE_prophet_h_8_1 :", mae_pro_h_8_1)
             result_pro_h_8_1 = pearsonr(réel_h_1["MW"], pred_pro_h_8_1["MW"])
             pv_pro_h_8_1 = result_pro_h_8_1[1]
             st.write("P-value PROPHET_h_8_1 :", pv_pro_h_8_1)

        if st.checkbox("Afficher la courbe et le scatter prédiction versus réalité pour PROPHET_h_8_1"):
             fig = plt.figure(figsize = (14, 7))
             plt.plot(réel_h_1, color = 'black', label = "Réalité")
             plt.plot(pred_pro_h_8_1, color='yellow', linestyle = ":", label = "Prédiction PROPHET_h_8_1") 
             plt.title('Entraînement sur 2013-2021 et prédiction horaire PROPHET de consommation sur 2021-2022 versus Réalité')
             plt.ylabel("Consommation en MW")
             plt.xticks(np.arange(0, 8736, 1440), ["mars 2021", "mai 2021", "juillet 2021", "septembre 2021", "novembre 2021", "janvier 2022", "mars 2022"])
             plt.legend()
             plt.show()
             st.pyplot(fig)

             fig = plt.figure(figsize = (14, 7))
             plt.scatter(réel_h_1, pred_pro_h_8_1, s= 2, color = "orange", label = "Prédiction PROPHET_h_8_1")
             plt.xlabel("Réalité")
             plt.ylabel("Prédictions")
             plt.plot([25000, 80000], [25000, 80000], color = 'black')
             plt.title("Consommation en MW : Prédictions de PROPHET_j_8_1 VS Réalité")
             plt.legend()
             st.pyplot(fig)
           
        
  if st.checkbox('Modèles de régression'):
    st.write("### Modèles de régression")
    st.write("(Sur le dataframe au pas journalier, du 01/03/2021 au 28/02/2022, avec la température ajoutée)")

    st.write("#### Entraînement des modèles et résultats")


    st.write("#### Application des modèles")
    st.write("#### Application des modèles")
    feats = pd.read_csv('feats.csv',index_col = 0)
    target = pd.read_csv('target.csv',index_col = 0)
    X_train,X_test, y_train, y_test= train_test_split(feats,target, test_size = 0.25, random_state=42)

   
  
    if st.checkbox("Entrainement et affichage des scores") :
        from sklearn.svm import SVR
        svr = SVR( kernel = 'linear' , gamma = 'auto')
        svr.fit( X_train, y_train)
        #Export du modèle chargé

        pickle.dump(svr, open("svr_loaded", 'wb'))


# SUR AUTRE .py
        # Récupération du modèle chargé
        #svr_loaded = pickle.load(open("svr_loaded", 'rb'))
        #svr_score_train = svr_loaded.score(X_train, y_train)
        #svr_score_test = svr_loaded.score(X_test, y_test)
        #st.write("Score de training : ", svr_score_train)
        #st.write("Score de training : ", svr_score_test)