import streamlit as st
import statsmodels.api as sm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
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
import pickle

#entête
image = Image.open('entete_en.png')
st.image(image, use_column_width=True, width=500)
st.markdown(
"<h2 style='text-align: center'>"
"<strong>La consommation électrique française </strong>"
"<br><span style='font-size: smaller'>de mars 2021 à février 2022</span>"
"</h2>"
, unsafe_allow_html=True)

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
#pred_pro_h_8_1 = pd.read_csv("pred_pro_h_8_1.csv", index_col=0)
réel_h_1 = pd.read_csv("réel_h_1.csv", index_col=0)
#pred_pro_8_1 = pd.read_csv("pred_pro_8_1.csv", index_col=0)
réel_1 = pd.read_csv("réel_1.csv", index_col=0)
df_compar_modèles = pd.read_csv("df_compar_modèles.csv", index_col=0)
df_metrics_machine_learning = pd.read_csv("df_metrics_machine_learning.csv", index_col=0)

######################
###Plan de l'appli####
######################
st.sidebar.subheader("Le réseau électrique français, data analyse de la consommation")
page1 = "Projet et résultats"
page2 = "Jeux de données"
page3 = "Visualisations"
page4 = "Prévision de la consommation"
page5 = "Corrections climatiques"
page6 = "Conclusion"


pages = [page1, page2, page3, page4, page5, page6]
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
  st.write("# Projet")
  st.write("## 1. Contexte")
  st.write("L’hiver 2022-2023, la guerre en Ukraine, ainsi que les maintenances prévues ou imprévues des centrales nucléaires françaises, ont amené pour la première fois depuis l’après-guerre  un risque important de sous-production de l’énergie par rapport à la consommation demandée. Au-delà de cette année exceptionnelle, de grandes transitions sont en cours et vont impacter très largement la production et la consommation énergétique dans la durée.")
  
  st.write("## 2. Objectif")
  st.write("Ce projet s'est focalisé sur l'analyse des données de RTE, le gestionnaire du réseau électrique français. L’électricité étant un « consommable » très peu stockable, le but est de développer des modèles de prévision de la consommation d'énergie électrique afin d’anticiper la production et d’éviter un black-out.")
  
  # afficher l'image au centre
  col1, col2, col3 = st.columns(3)

  with col1:
            st.write("")    
  with col2:
          image = Image.open('image blackout.jfif')
          st.image(image, use_column_width=True, width=500)

  with col3:
          st.write("")


  st.write("Le réseau français étant mutualisé, les analyses ont été faite principalement au niveau national et à l’échelle de la demi-heure, de l’heure ou de la journée.")
  
  
  st.markdown(
	"<div style='border: 1px solid rgba(0, 104, 201, 0.1); padding: 5px; font-style: italic;"
	"padding: 5px; font-style: italic; text-align: justify; background-color: rgba(0, 104, 201, 0.1); "
	"color: #1e6777; border-radius: 0.25rem; font-size: 15px'>"	
	"Pour accéder à nos analyses détaillées et interactives (Data visualisation, Machine learning...), cliquez dans le menu de gauche. "
	"Quant à nos conclusions, vous les trouverez ci-dessous ainsi qu'en dernière page."
	"</div>", unsafe_allow_html=True)
  
  
  st.write("# Résultats")
  st.write("La prévision de la consommation avec les modèles de régression (Linear Rgression, Decision Tree, Random Forest, KNN et SVM) a donné de trop bons résultats : les variables explicatives (les filères de production) sont liées à la variable cible.")
  st.write("Le modèle de Time Series SARIMAX, intégrant en variable exogène la température, a permis une bonne prédiction de la consommation selon le temps, sur 14 jours au pas journalier (RMSE = 1700).")
  st.write("La prévision de la consommation au pas horaire et/ou sur le long terme nécessite des modèles de séries temporelles prenant en compte des saisonnalités multiples :")
  st.write(" - pour une étendue de prédiction de 14 jours au pas journalier TBATS (RMSE = 3067) ou PROPHET intégrant la température en variable externe (RMSE = 3480) ;") 
  st.write(" - pour 1 an, PROPHET au pas journalier (RMSE = 4612) ou au pas horaire (RMSE = 4855).")
  st.write("L'utilisation de données de consommation déclimatisées n'a pour l'instant pas abouti à des résultats satisfaisants, mais constitue une piste à explorer potentiellement très intéressante.")

if select_page == page2 : 
  st.write("# Jeux de données")
  ques_machine=st.radio(" ",('Jeu de données principal', 'Jeu de données secondaire')) 
  if ques_machine =='Jeu de données principal':
    st.write("## 1. Source")
    st.write("Ce [jeu de données](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure) est librement accessible depuis la plateforme Open Data Réseaux Énergies.  Les données proviennent de l’historisation de l’application “éCO2 mix” créée par RTE, l’entreprise qui gère le réseau de l’électricité française.")
   
    st.write("## 2. Période")
    st.write("Les données sont enregistrées pour chaque région, toutes les 30 minutes, du 1er janvier 2013 au 26 avril 2022, soit 1 969 435 lignes.")
    
    st.write('## 3. Exploration des données')
    st.write("Langage utilisé : Python")
    #st.markdown("Langage utilisé : <br><strong> Python<strong>", unsafe_allow_html=True)
    st.write("Librairies utilisées : Pandas, Numpy, Matpltlib, Seaborn, Geopandas, Scikit-learn, Statsmodels.tsa.seasonal, sarimax, tbats, prophet (+ à vérifier)")
    st.write("Taille de DataFrame : 1 969 435  lignes X 32 colonnes.")
    
    st.markdown("<br><strong> Les variables</strong>", unsafe_allow_html=True)
    st.write("Le dataset compte 32 variables.")
    if st.checkbox("Afficher les 32 variables") :
      st.image(Image.open('liste_var.png'))
      
    st.write("La variable cible est « Consommation (MW) ».")
    
      

    st.write("## 4. Traitement des données")
    st.write("Le jeu de données est à 52% vide. Les 14 premières variables sont très complètes : 93% de complétion. Les filières de productions nucléaire et pompages ont des données vides car elles ne sont pas présentes dans 5 des régions. Ces valeurs pourront être remplacées par 0. Par contre, les 17 dernières colonnes sont très parcellaires (18% de complétion).")
    st.write("La puissance (en MW ici) est une mesure instantanée (comme un flux). Les puissances instantanées produites par une même filière à différents moments se combinent en faisant leur moyenne, alors que les puissances produites par les différentes filières doivent se sommer pour obtenir la puissance totale produite, de même on doit sommer les consommations et les productions des différentes régions pour avoir les totaux. La stratégie énergétique de la France étant centralisée et les régions interconnectées, la majorité des analyses seront au niveau national.")
    st.write("Pour prendre en compte la puissance de calcul, le périmètre de l'étude est réduit à une année, du 1er mars 2021 au 28 février 2022. De même la majorité du temps, les données seront au pas journalier.")
    st.write("Afin d’alléger le futur modèle, certaines variables sont supprimées. La sélection se fait en fonction du taux de complétion,  des valeurs majoritairement nulles (ou très faible).")
    st.write("La « Date » est passé au format datetime.")
    
    st.write('## 5. Ajout de variables')
    
    st.write("Pour notre dataframe principal, 4 variables ont été ajoutées.")
    st.write("Décomposition de la  date : « mois », « jour » et « WE_et_JF » (0 pour un jour de semaine classique, 1 pour un jour de week-end ou un jour férié).")
    st.write("Météorologique : « température » pour la température moyenne journalière nationale en °C (voir données secondaires)")
    
    st.markdown("<br><strong> Extrait du DataFrame et ses dimensions</strong>", unsafe_allow_html=True)
    st.dataframe(dfMJ.iloc[:,1:16].head(10))
    
    st.write(dfMJ.shape)
 
    
    
  if ques_machine =='Jeu de données secondaire':
    st.write("## 1. Source")
    st.write("Ce [jeu de données](https://odre.opendatasoft.com/explore/dataset/temperature-quotidienne-departementale/table/?disjunctive.departement&sort=date_obs&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJhcmVhcmFuZ2UiLCJmdW5jIjoiQ09VTlQiLCJ5QXhpcyI6InRtb3kiLCJzY2llbnRpZmljRGlz) est librement accessible depuis la plateforme Open Data Réseaux Énergies. Il permet d'obtenir les températures moyennes départementales récoltées par weathernews.fr.")
    


if select_page == page3 : 
  st.write("# Visualisation")

  st.markdown(
	"<div style='border: 1px solid rgba(0, 104, 201, 0.1); padding: 5px; font-style: italic;"
	"padding: 5px; font-style: italic; text-align: justify; background-color: rgba(0, 104, 201, 0.1); "
	"color: #1e6777; border-radius: 0.25rem; font-size: 15px'>"	
	"Afin de faciliter la visualisation des regroupements ont été créés."
    " Nous avons les variables «Production_MW», «Production_verte_MW» et «Differentiel_MW» . "
	"</div>", unsafe_allow_html=True)

  

  temp1 = "Courbes générales"
  temp2 = "Périodicité"
  temp3 = "Température"
  temp4 = "Géographie"
  dataviz_temp = st.radio("", (temp1, temp2, temp3, temp4))
  
  #Courbe générale
	################
  if dataviz_temp == temp1:
      
      
    st.write("Observons l'évolution des principales variables dans le temps.")
    liste_var = st.multiselect('Sélectionnez les variables :',
			['Consommation_MW', 'Production_MW',"Production_verte_MW", "Differentiel_MW"],
	       default=['Consommation_MW', 'Production_MW', "Production_verte_MW"
			     ])
    fig=px.line(data_frame=dfMW_temp_nationale_date_somme, x='Date', y=liste_var, title= str(liste_var) + "par date")
    st.plotly_chart(fig)  
    
    st.write("On note un lien entre les courbes de production et consommation au cours de l’année. Il y a un effet de la saison avec une courbe de consommation beaucoup plus basse de mai à octobre et beaucoup plus haute en janvier et février. "
    "La production fluctue également, mais dans des limites plus étroites, et baisse proportionnellement moins en été, ce qui induit un différentiel plus important durant cette saison.")
    st.write("On remarque également des fluctuations de ces courbes sur un laps de temps plus court, de l’ordre de la semaine.")
    st.write("La courbe de la production “verte” varie dans une moindre limite, avec des fluctuations plus importantes en amplitude (MW) d’octobre à juin.")
    
    
    st.write("Observons la relation entre les différentes variables.")
    liste_coul = st.selectbox('Sélectionnez la variable de coloration :',
			["Production_verte_MW", "Differentiel_MW"],)
    fig5=px.scatter(data_frame=dfMW_temp_nationale_date_somme, y='Consommation_MW', x='Production_MW', color=liste_coul , title = 'Etude de corrélation entre la production et la consommation en fonction du/de la ' + str(liste_coul))
    st.plotly_chart(fig5)
    st.write("La consommation et la production sont corrélées (confirmé par un test de Pearson à un seuil de 5%). La production verte est plus importante quand la production totale est importante et excédentaire par rapport à la consommation.")
  
    
    if st.checkbox("Afficher la distribution de la consommation") :

     fig, ax = plt.subplots()
     n_bins = st.slider(
        label = "Choisissez un nombre de bins",
        min_value=10,
        max_value=100,
        value=20)
     fig = plt.figure()
     plt.hist(dfMW_nationale_date_somme['Consommation_MW'],bins=n_bins, label=["Consommation (MW)"], rwidth = 0.9)
     plt.legend()
     plt.title('Distribution de la consommation nationale par demi-heure')
     st.pyplot(fig)
     st.write("La consommation électrique totale nationale varie entre 29 660 MW (le 21/08/2021 à 7h00) et 83 333 MW (le 29/11/2021 à 19h00). La valeur la plus fréquente est environ 50 000 MW (49 756 MW).")
     st.write("Elle suit une distribution proche de la normale.")
    

    
  elif dataviz_temp == temp2:
      periodi1 = "Mensuelle et journalière"
      periodi2 = "Horaire"
      select_periodi = st.radio("", (periodi1, periodi2))
      if select_periodi == periodi1:

          image = Image.open('mois et jour.png')
          st.image(image, use_column_width=True, width=500)
          st.write("La consommation est plus élevée sur les premiers et derniers mois (donc en hiver) et plus faible en milieu d’année donc en été (constat visuel confirmé par un test de Pearson à 5%).")
          st.write("De plus, la moyenne de la puissance les jours de week-end et jours fériés est toujours plus élevée que celle des jours de semaine (également confirmé par un test de Pearson à 5%).")
          
     
      if select_periodi == periodi2:
          image = Image.open('heure.png')
          st.image(image, use_column_width=True, width=500)
          
          st.write("La consommation est également corrélée à l’heure de la journée (confirmé par Pearson).")
          st.write("Il y un creux pendant la nuit autour 4h30 et des pics autour  de 13h00 et à nouveau autour de 19h30 (coïncidant avec la préparation des repas chauds et les retour au domicile). ")

  
  elif dataviz_temp == temp3:
    fig4=px.scatter(data_frame=dfMW_temp_nationale_date_somme, y='Consommation_MW', x='temperature',  title = 'Etude de corrélation entre la consommation journalière nationale moyenne et la température')
    st.plotly_chart(fig4) 
    st.write("La corrélation est confirmé par Pearson.")
    st.write("De 0 à 13°C la consommation augmente fortement au fur et à mesure que la température baisse, probablement à cause de l’utilisation de chauffage électrique. Ce mode de chauffe est présent dans 35% des foyers et dans ce cas représente 67% de la consommation du foyer.")
    st.write("A partir de 13°C  la consommation stagne ou augmente légèrement, probablement car le chauffage n’est plus utilisé mais il peut y avoir une utilisation de la climatisation (dont l’équipement est moins fréquent en France)")
    

  if dataviz_temp == temp4:
      image = Image.open('conso en fonction hab.png')
      st.image(image, use_column_width=True, width=500)
      st.write("La consommation est corrélée au nombre d'habitants de chaque région (confirmation visuelle et statistique).")
      st.write("L’Ile-de-France est la région avec la plus grande consommation mais aussi le plus grand nombre d’habitants. Elle semble en dessous de la moyenne nationale.")
      
      
      image = Image.open('carte conso pour 1000 bleu.png')
      st.image(image, use_column_width=True, width=500)
      st.write("A l’inverse, les Hauts-de-France, la Normandie et l’Auvergne-Rhône-Alpes les ratios les plus élevés. Plusieurs pistes pourraient expliquer cette différence régionale : la part de l’industrie, les typologies des bâtiments  (surface habitable par personne, mode de chauffage …).")


  if st.checkbox("Afficher le focus sur la production") :
      image = Image.open('prod filière.png')
      st.image(image, use_column_width=True, width=500)
      st.write("La filière nucléaire est largement prédominante dans le mix électrique français puisqu’elle représente presque 70% de toute la production nationale sur notre année d’observation.")
      st.write("Le thermique également pilotable représente 7.5% de la production électrique.")
      st.write("En 2021-2022, moins d’un quart de la puissance provient d’une filière d’électricité verte (hydraulique + éolien + solaire + biothermique).")

if select_page == page4 : 
  st.write("# Prévision de la consommation")
  #st.markdown("<h3 style='text-align: center; color: grey;'>Prévision de la consommation électrique française <br>  </h3>", unsafe_allow_html=True)
  st.write("Type de machine learning") 
  if st.checkbox('Modèles de régression'):
    st.write("### Modèles de régression")
    st.write("(Sur le dataframe au pas journalier, du 01/03/2021 au 28/02/2022, avec la température ajoutée)")

    st.write("#### Entraînement des modèles et résultats")


    st.write("#### Application des modèles")
    feats = pd.read_csv('feats.csv',index_col = 0)
    target = pd.read_csv('target.csv',index_col = 0)
    X_train,X_test, y_train, y_test= train_test_split(feats,target, test_size = 0.25, random_state=42)
    ques_machine=st.radio("Choix du modèle de machine learning", ('Régression','Arbre de décision','Random Forest', 'KNN', 'SVR'))
    if ques_machine == 'Régression':
      if st.checkbox("Entrainement et affichage des scores") :
        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(X_train, y_train)
        lr_score_train = regressor.score(X_train, y_train)
        lr_score_test = regressor.score(X_test, y_test)
        st.write("Score de training : ", lr_score_train)
        st.write("Score de test : ", lr_score_test)
    if ques_machine == 'Arbre de décision':
      if st.checkbox("Entrainement et affichage des scores") :
        from sklearn.tree import DecisionTreeRegressor
        regressor_dt=DecisionTreeRegressor(random_state = 42)
        regressor_dt.fit(X_train, y_train)
        dtr_score_train = regressor_dt.score(X_train, y_train)
        dtr_score_test = regressor_dt.score(X_test, y_test)
        st.write("Score de training : ", dtr_score_train)
        st.write("Score de test : ", dtr_score_test)
    if ques_machine == 'Random Forest':
      if st.checkbox("Entrainement et affichage des scores") :
        from sklearn.ensemble import RandomForestRegressor
        regressor_rf = RandomForestRegressor(random_state=42)
        regressor_rf.fit(X_train, y_train)
        rfr_score_train = regressor_rf.score(X_train,y_train)
        rfr_score_test = regressor_rf.score(X_test,y_test)
        st.write("Score de training : ", rfr_score_train)
        st.write("Score de test : ", rfr_score_test) 
    if ques_machine == 'KNN':  
       if st.checkbox("Entrainement et affichage des scores") :
        from sklearn.neighbors import KNeighborsRegressor
        knr = KNeighborsRegressor()
        knr.fit(X_train, y_train)
        knr_score_train = knr.score(X_train, y_train)
        knr_score_test = knr.score(X_test, y_test)
        st.write("Score de training : ", knr_score_train)
        st.write("Score de test : ", knr_score_test)       
    if ques_machine == 'SVR':
       if st.checkbox("Entrainement et affichage des scores") :
        from sklearn.svm import SVR
        # Récupération du modèle chargé
        svr_loaded = pickle.load(open("svr_loaded", 'rb'))
        svr_score_train = svr_loaded.score(X_train, y_train)
        svr_score_test = svr_loaded.score(X_test, y_test)
        st.write("Score de training : ", svr_score_train)
        st.write("Score de test : ", svr_score_test)



    st.write("#### Récapitulatif des métriques des algorithmes de régression") 
    if st.checkbox("Afficher le tableau des métriques des modèles de régression") :
      st.write(df_metrics_machine_learning)

    st.write("#### Importance des variables")
    if st.checkbox("Afficher l'importance des variables") :
      image = Image.open('importance var.png')
      st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


    st.write("#### Conclusions des modèles de régression")
    st.write("Les prédictions des modèles sont trop bonnes. Explication possible : la variable cible doit être liée aux variables explicatives initiales. Nous allons donc par la suite tenter de prédire la consommation en fonction du temps. De plus, nous avons pu vérifier que la température influe énormément sur la consommation.")    
    
  if st.checkbox('Modèles de séries temporelles'):

     st.write("### Modèles de séries temporelles")
     st.write("#### Introduction")
     st.write("Les séries temporelles sont « une suite d’observations chiffrées d’un même phénomène, ordonnées dans le temps ». Ce type de série se retrouve dans de nombreux domaines comme l'économie, l'environnement ou encore la santé.")
     st.write("L'intérêt pour la prévision par des séries temporelles du comportement futur de la consommation électrique est de pouvoir se détacher des variables de production, qui sont liées à la variable cible 'consommation'.")
     st.write("Les modèles prédisant la consommation dans le temps reposent sur différents algorithmes ; le tableau ci-dessous récapitule leurs principales caractéristiques pratiques en regard de notre problématique.")

     if st.checkbox("Afficher la démarche") :
       st.write("Le modèle de base SARIMA a tout d'abord été entraîné sur la période de référence de 1 an, en lui faisant prédire les 14 derniers jours.")
       st.write("La variable exogène 'Température', dont l'importance était apparue suite aux résultats des modèles de régression, a ensuite été intégrée au modèle. Au vu des bonnes capacités de calcul du modèle, la période d'entraînement ainsi que celle de prédiction ont été étendues.")
       st.write("Il est apparu que SARIMAX n'intégrait pas les saisonnalités multiples (à la journée, à la semaine et à l'année), ce qui rendait difficiles les prédictions horaires sur 14 jours (saisonnalités à la journée et à la semaine) et à long terme sur 1 an.")
       st.write("Des modèles tels que TBATS et PROPHET ont donc enfin été appliqués.")

     st.write("#### Comparatif des modèles de Time Series") 
     st.write(df_compar_modèles)
     st.write("Comme il l'a été brièvement vu dans le tableau comparatif, chacun de ces modèles a son domaine d'excellence en matière de prédiction des consommations. Toutefois, dans une perspective métier, il semble intéressant d'être capable de produire de bonnes prédictions au pas horaire (et non simplement journalier) et de pouvoir tenir compte de la température.")
     st.write("Il n'a pas été possible, dans le temps imparti pour ce projet, d'obtenir et d'utiliser des températures au pas horaire.")
     st.write("Il serait intéressant de pouvoir tester PROPHET au pas horaire avec les températures en variable exogène, ainsi que TBATS en utilisant des consommations déclimatisées.")
 

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
             
             #Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Pickle :
             m_pro_j_loaded = pickle.load(open("m_pro_j_loaded", 'rb'))

             # Prévision dans un df des dates futures, à l'heure (365 jours x 24 heures = 8760 pas horaires):

             future = m_pro_j_loaded.make_future_dataframe(periods=365, include_history = False)

             # prédiction (df)
             pred_pro_j_8_1 = m_pro_j_loaded.predict(future)

             # df des éléments de la prédiction
             pred_pro_j_8_1 = pred_pro_j_8_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

             pred_pro_j_8_1.index = réel_1.index
             pred_pro_j_8_1 = pred_pro_j_8_1["yhat"]

             mae_pro_j_8_1 = mean_absolute_error(réel_1, pred_pro_j_8_1)
             rmse_pro_j_8_1 = mean_squared_error(réel_1, pred_pro_j_8_1, squared=False)
             st.write("RMSE_prophet_j_8_1 :", rmse_pro_j_8_1)
             st.write("MAE_prophet_j_8_1 :", mae_pro_j_8_1)
             result_pro_j_8_1 = pearsonr(réel_1["Consommation (MW)"], pred_pro_j_8_1)
             pv_pro_j_8_1 = result_pro_j_8_1[1]
             st.write("P-value PROPHET_j_8_1 :", pv_pro_j_8_1)
             


        if st.checkbox("Afficher la courbe et le scatter prédiction versus réalité pour PROPHET_8_1"):
             fig = plt.figure(figsize = (14, 7))
             plt.plot(réel_1, color = 'black', label = "Réalité") # l'année' de vraie conso
             plt.plot(pred_pro_j_8_1, color='orange', linestyle = "--", label = "Prédiction PROPHET") # la prédiction PROPHET de conso sur l'année
             plt.title('Entraînement sur 2013-2021 et prédiction journalière PROPHET de consommation sur 2021-2022 versus Réalité')
             plt.ylabel("Consommation en MW")
             plt.xticks(np.arange(0, 364, 60), ["mars 2021", "mai 2021", "juillet 2021", "septembre 2021", "novembre 2021", "janvier 2022", "mars 2022"])
             plt.legend()
             plt.show()
             st.pyplot(fig)

             fig = plt.figure(figsize = (14, 7))
             plt.scatter(réel_1, pred_pro_j_8_1, color = "orange", label = "Prédiction PROPHET")
             plt.xlabel("Réalité")
             plt.ylabel("Prédictions")
             plt.plot([25000, 80000], [25000, 80000], color = 'black')
             plt.title("Consommation en MW : Prédictions de PROPHET_j_8_1 VS Réalité")
             plt.legend()
             st.pyplot(fig)
        

     if ques_machine =='1 an / Pas horaire':
        if st.checkbox("Entraînement du modèle et affichage des métriques pour PROPHET_h_8_1") :
              
             

              #Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Pickle :
              m_pro_h_loaded = pickle.load(open("m_pro_h_loaded", 'rb'))

              # Prévision dans un df des dates futures, à l'heure (365 jours x 24 heures = 8760 pas horaires):

              future = m_pro_h_loaded.make_future_dataframe(periods=8760, freq = "h", include_history = False)

              # prédiction (df)
              pred_pro_h_8_1 = m_pro_h_loaded.predict(future)

              # df des éléments de la prédiction
              pred_pro_h_8_1 = pred_pro_h_8_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

              pred_pro_h_8_1.index = réel_h_1.index
              pred_pro_h_8_1 = pred_pro_h_8_1["yhat"]

              mae_pro_h_8_1 = mean_absolute_error(réel_h_1, pred_pro_h_8_1)
              rmse_pro_h_8_1 = mean_squared_error(réel_h_1, pred_pro_h_8_1, squared=False)
              st.write("RMSE_prophet_h_8_1 :", rmse_pro_h_8_1)
              st.write("MAE_prophet_h_8_1 :", mae_pro_h_8_1)
              result_pro_h_8_1 = pearsonr(réel_h_1["MW"], pred_pro_h_8_1)
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
           
        

     st.write("#### Récapitulatif des courbes de prédiction des meilleurs modèles")
     ques_machine=st.radio("Choix de l'étendue de prévision", ('14 jours', '1 an')) 
     if ques_machine =='14 jours':
       if st.checkbox("Au pas journalier à 14 jours") :
          image = Image.open('courbes_j_14.png')
          st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
          st.write("Le tracé des courbes de prédiction suit globalement celui de la courbe de consommation réelle, tout en la surestimant. Les prévisions du modèle SARIMAX intégrant la température en facteur exogène sont les plus proches de la réalité.")
       if st.checkbox("Au pas horaire à 14 jours") :
          image = Image.open('courbe_tb_h_1_14_suite.png')
          st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
          st.write("La courbe de prédiction du modèle TBATS suit bien celle de la consommation réelle, tout en la sous-estimant néanmoins du 6e au 8e jour.")       
        
     if ques_machine =='1 an':
        if st.checkbox("Au pas journalier à 1 an") :
          image = Image.open('courbes _j_3_meilleurs_1.png')
          st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
          st.write("Au contraire de SARIMAX dont la courbe de prédiction finit par 'décrocher' de celle de la consommation réelle, les modèles TBATS et PROPHET intègrent bien les effets de la double saisonnalité (sur la semaine et sur l’année) ; les courbes de prédiction suivent bien les valeurs réelles, tout en les sous-estimant pour cette année-là. PROPHET est meilleur que TBATS sur ces prévisions à 1 an.")
        if st.checkbox("Au pas horaire à 1 an") :
          image = Image.open('courbe_pro_h_1.png')
          st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
          st.write("La courbe de prédiction du modèle PROPHET sous-estime la réalité des trois premiers et des trois derniers mois, mais suit remarquablement la courbe de la consommation réelle de juillet à novembre. Le modèle doit intégrer ici une triple saisonnalité : à la journée, à la semaine et à l'année.")

     st.write("#### Tableau récapitulatif des métriques des modèles")
     if  st.checkbox("Afficher le tableau des métriques") :
        st.write(df_métriques_sans_reclim)
     
     st.write("La fréquente sous-estimation des valeurs de consommation sur une prévision à l'année nous a conduit à nous interroger sur l'impact des températures sur la variable cible. Effectivement, il semble que notre période à prédire (2021-2022) soit atypique en regard des années d'entraînement qui la précèdent. Un argument allant dans ce sens est le meilleur résultat obtenu par PROPHET sur la prédiction de l'année 2019 (voir tableau des métriques).")
     st.write("Il est dès lors apparu important de pouvoir limiter les effets, sur la prédiction de la consommation, des fluctuations des températures réelles par rapport aux normales saisonnières.") 

     
if select_page == page5 : 
    st.write("# Corrections climatiques")

    st.write("L'année 2021 a été plus froide que les 7 années précédentes. Nos prévisions de consommation sur 2022 ont donc été faites sous l'influence de températures (impactant les consommations) qui ne sont pas identiques aux températures normales (au sens de normales saisonnières).")
    st.write("Pour dépasser cette limite, nous avons entrepris une approche mathématique visant à entraîner nos modèles avec des données de consommation déclimatisées. Nous avons ainsi obtenu des prédictions déclimatisées et avons exploré la possibilité de les reclimatiser en fonction des températures attendues à l'horizon de 14 jours.")
    st.write("## Théorie")  # et/ou checkbox "Afficher la théorie"
    
    
    st.write("### Déclimatisation")
    st.markdown("<br><strong> Objectif intermédiaire </strong>", unsafe_allow_html=True)
    st.write("Obtenir des données de consommations déclimatisées pour entrainer les modèles")
    
    st.write("La fonction consommation est :")
    st.markdown(
  	"<div style='border: 1px solid rgba(0, 104, 201, 0.1); padding: 5px; font-style: italic;"
  	"padding: 5px; font-style: italic; text-align: justify; background-color: rgba(0, 104, 201, 0.1); "
  	"color: #1e6777; border-radius: 0.25rem; font-size: 15px'>"	
  	"Soit : Consommation : C, Température : temp et Temps : t . "
    "\n\n"
    "Tels que A et B sont les paramètres de la régression"
  	"</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>C(temp,t)=A * temp+B</h3>", unsafe_allow_html=True)


    #st.write("tels que A et B sont les paramètres de la régression.")
    
    st.write("La consommation est corrélée à la température à partir d'une température moyenne journalière de  13,7°C.")
    image = Image.open('pvalue_temperature.png')
    st.image(image, use_column_width=True, width=500)
    st.write("Ainsi, notre formule précédente se transforme en :")
    st.write("Pour temp>13.7°C : C_sup(t)=B(t) ")
    st.write("Pour temp<13.7°C : C_inf(temp,t)=A * temp+B")
    st.markdown(
  	"<div style='border: 1px solid rgba(0, 104, 201, 0.1); padding: 5px; font-style: italic;"
  	"padding: 5px; font-style: italic; text-align: justify; background-color: rgba(0, 104, 201, 0.1); "
  	"color: #1e6777; border-radius: 0.25rem; font-size: 15px'>"	
  	"Donc pour C0_inf(t) : consommation déclimatisée"
  	"</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>C0_inf(t)= C_inf(temp,t)−A(temp−temp_normal)</h3>", unsafe_allow_html=True)
    st.write("En appliquant le modèle de régression sur nos consommations à températures inférieures, on trouve un coefficient directeur de régression égal à : -2501,3 MW/°C.")
    st.markdown("<h3 style='text-align: center;'>C0_inf(t) = C_inf(temp,t) + 2501.3  * (temp−temp_normal)</h3>", unsafe_allow_html=True)
    
    
    st.write("Les températures normalisées sont disponibles sur le site d'[enedis](meet.google.com/wkw-gukz-dsu).")
    st.write("Une fois la courbe de consommation d'entrainement obtenus, les séries temporelles sont utilisées pour obtenir des prévisions déclimatisées.")
        
    #if st.checkbox('Reclimatisation'):
    st.write("### Reclimatisation")  
    st.markdown("<br><strong> Objectif intermédiaire : </strong>", unsafe_allow_html=True)
    st.write("Reclimatiser les prévisions de consommations en fonction des écarts entre les températures réelles (ou prévisibles) et les normales.")
    
    
    
    st.markdown(
      	"<div style='border: 1px solid rgba(0, 104, 201, 0.1); padding: 5px; font-style: italic;"
      	"padding: 5px; font-style: italic; text-align: justify; background-color: rgba(0, 104, 201, 0.1); "
      	"color: #1e6777; border-radius: 0.25rem; font-size: 15px'>"	
         " Soit : Delta_C est la variation de consommation dû au différentiel de température "
         "\n\n"
          "Tn : la température normalisée, Tr : la température réelle (à prédire)"
          "\n\n"
          "k : le coeeficient de reclimatisation des prévisions = -2501.3 MW/°C"
      	"</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Delta_C (en MW) = -k * (Tn - Tr)</h3>", unsafe_allow_html=True)
            
    st.write("Afficher Delta de façon spécifique")
                
            # Sliders pour Tn et Tr
            # Diviser l'écran en deux colonnes
    col1, col2 = st.columns(2)

            # Sliders pour Tn et Tr
    with col1:
                Tn = st.number_input("Température normale : Tn (°C)", 0.0, 20.0, 10.5, step = 0.1)

    with col2:
                Tr = st.number_input("Température réelle : Tr (°C)", 0.0, 20.0, 15.0, step =0.1)

          
    def calculer_delta(Tn, Tr):
                #k=-2501.3
            if Tn < 13.7 and Tr < 13.7:
                return "-k * (Tn - Tr)"
            elif Tn >= 13.7 and Tr >= 13.7:
                return 0
            elif Tn >= 13.7 and Tr < 13.7:
                return "-k * (13.7 - Tr)"
            elif Tn < 13.7 and Tr >= 13.7:
                return "-k * (Tn - 13.7)"

        # Calculer delta
    delta = calculer_delta(Tn, Tr)

            # Afficher le résultat
    st.markdown(f"<h3 style='text-align: center;'>Delta_C (en MW) = {delta}</h3>", unsafe_allow_html=True)
            
            #afficher le résultat numérique en remplaçant k
    st.write("Faire le calcul en remplaçant k :")
    def calculer_deltak(Tn, Tr):
                k=-2501.3
                if Tn < 13.7 and Tr < 13.7:
                    return round (-k * (Tn - Tr),1)
                elif Tn >= 13.7 and Tr >= 13.7:
                    return 0
                elif Tn >= 13.7 and Tr < 13.7:
                    return round (-k * (13.7 - Tr), 1)
                elif Tn < 13.7 and Tr >= 13.7:
                    return round ( -k * (Tn - 13.7), 1)

            # Calculer delta
    deltak = calculer_deltak(Tn, Tr)

            # Afficher le résultat
    st.markdown(f"<h3 style='text-align: center;'>Delta_C (en MW) = {deltak}</h3>", unsafe_allow_html=True)
            


    st.write("## Résultats des prédictions reclimatisées")
    st.write("Les performances de ces prévisions reclimatisées sont : ")
    st.write(df_métriques_reclim)
    
    image = Image.open('courbes_toutes_reclim.png')
    st.image(image, use_column_width=True, width=500)
    st.write("Les résultats de prédiction des trois modèles sont non significatifs (cf p-values). On voit sur le graphique qu'ils surestiment la réalité et ne reproduisent que grossièrement la forme de la courbe de consommation réelle.")
    st.write("Les prédictions sont moins bonnes que celles des modèles utlisant les données de consommation non 'déclimatisées'. C'est particulièrement visible sur ce graphique si l'on compare la courbe de SARIMA_reclimatisé et celle de SARIMAX_temp.")
    st.write("L’hypothèse est que notre coefficient de régression k unique est un coefficient moyen et qu’il faudrait donc calculer un coefficient pour chaque pas de la période à prédire (une différence de 2°C n’a pas forcément le même impact sur la consommation entre 13 et 11°C qu’entre 10 et 8°C, par exemple).")
    
    st.write("#### Perspectives") # et/ou checkbox "Afficher les perspectives"
    st.write("Pour aller plus loin, le coefficient de thermosensibilité doit être calculé par pas demi horaire (soit 17 520 coefficients pour une année : 17 520 = 24 x 2 x 365). ")
    st.write("Pour pouvoir obtenir cette matrice de valeurs de coefficients de sensibilité à la température, nous partons de trois années d’observations réalisées de la consommation et de température. Pour chaque pas demi horaire, on aura trois consommations et trois températures observées, on appliquera sur ces deux trios une régression qui va nous permettre d’identifier un coefficient de sensibilité exprimé en MW/°C.")
    st.write("En appliquant le même algorithme sur le reste des identifiants demi horaires, nous obtiendrons notre matrice de sensibilité à la température. On utilisera cette matrice pour déclimatiser nos observations (consommations déclimatisées d’entraînement des modèles) et reclimatiser nos prédictions.")

if select_page == page6 : 
    st.write("#### Conclusion générale")


   
    st.write("Afin d’éviter des black-out liés au déphasage entre la production et la consommation, cette étude s'est concentrée sur la prédiction de la consommation car c'est la variable la moins pilotable. "
             "Les apports/enseignements/résultats de cette étude sont :")
    
    st.write("- La consommation est notamment corrélée à des variables temporelles et à la température.")
    st.write("- La prévision de la consommation avec les modèles de régression a donné de trop bons résultats : les variables explicatives (les filères de production) sont liées à la variable cible.")
    st.write("- Les séries temporelles sont d’excellents outils pour faire ces prédictions. Chaque modèle a produit des résultats significatifs mais a chacun son domaine d'application préférentiel."
    " Par exemple : pour une prédiction sur 14 jour au pas horaire, nous conseillons TBATS ; pour une prédiction sur 1 an mieux vaut privilégier Prophet.")
    st.write("- L'utilisation de données de consommation déclimatisées n'a pour l'instant pas abouti à des résultats satisfaisants, mais constitue une piste à explorer potentiellement très intéressante.")


    st.markdown(
	"<div style='border: 1px solid rgba(0, 104, 201, 0.1); padding: 5px; font-style: italic;"
	"padding: 5px; font-style: italic; text-align: justify; background-color: rgba(0, 104, 201, 0.1); "
	"color: #1e6777; border-radius: 0.25rem; font-size: 15px'>"	
	"Remerciements : nous tenons à remercier Manon, de DataScientest, pour sa compétence et sa disponibilité tout au long de ce projet."
	"</div>", unsafe_allow_html=True) 
