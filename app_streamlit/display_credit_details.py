import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
import pickle
from urllib.request import urlopen
import json
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(layout="wide")

def main() :
    
    @st.cache
    def load_data():
    
        '''Récupération des données'''
    
        # Données générales
        z = ZipFile('data/general_info_app.zip')
        initial_data = pd.read_csv(z.open('general_info_app.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    
        # Données de travail pour sélection modèle
        z = ZipFile('data/training_data_clean.zip')
        clean_data = pd.read_csv(z.open('training_data_clean.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
        target = clean_data['TARGET']
        #clean_data = clean_data.drop(columns='TARGET', axis=1)
    
        # Description
        description = pd.read_csv("data/features_description.csv", 
                               usecols=['Row', 'Description'], encoding= 'unicode_escape')


        return initial_data, clean_data, target, description 
    
    def load_model():

        ''' Récupération du modèle entrainé '''
        
        pickle_in = open('model/LGBM_model_final.pkl', 'rb') 
        clf = pickle.load(pickle_in)
    
        return clf

    
    def gauge_plot(score):
    
        fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = score,
        mode = "gauge+number",
        title = {'text': "Score client"},
        delta = {'reference': 0.3},
        gauge = {'axis': {'range': [None, 100]},
                 'bar' : {'color' : '#a21251'},
             'steps' : [
                 {'range': [0, 30], 'color': "#d4e6fa"},
                 {'range': [30, 60], 'color': "#7fb6f1"},
                 {'range': [60, 100], 'color': "#2b86e9"} ],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 30}}))

        st.plotly_chart(fig)
        
    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn


    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data, customer_id):
        data_client = data[data.index == int(customer_id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"].abs() /365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income


    @st.cache
    def load_kmeans(sample, customer_id, mdl):
        index = sample[sample.index == int(customer_id)].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data_client], axis=1)
        return df_neighbors.iloc[:,1:].sample(10)

    @st.cache
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 

    def proba_load(customer_id):
        
        #API_url = "http://127.0.0.1:5000/get_score_credit/" + str(customer_id)
        API_url = "https://get-scoring-credit.herokuapp.com/get_score_credit/" + str(customer_id)
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        pred_proba = API_data['score']
        
        return pred_proba        
        
        
        
    # Chargement des données
    initial_data, clean_data, target, description = load_data()
    model = load_model()
    customer_id = clean_data.index.values
    data_wk = clean_data.copy()
    data_wk['PRED_PROBA'] = model.predict_proba(data_wk.drop(columns='TARGET', axis=1))[:,1]
    


    #######################################
    #               SIDEBAR               #
    #######################################

    # TITRE
    html_temp = """
    <div style="background-color: grey; padding:10px; border-radius:10px">
    <h1 style="color: black; text-align:center">PRÊT À DÉPENSER - Score crédit dashboard</h1>
    </div>
    <p style="font-size: 20px; font-weight: italic; text-align:left"></p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    #st.title('Prêt à dépenser - Score crédit dashboard')

    # Selection des identifiants clients
    st.sidebar.header("** Informations générales **")

    #Loading selectbox
    selected_id = st.sidebar.selectbox("Client ID", customer_id)

    # Chargement des informations générales (le nombre de crédit dans l'échantillon, les revenus moyens et le montant emprunté moyen, pourcentage en défaut )
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(clean_data)


    ### Affichage des informations dans la sidebar ###
    # Nombre total de crédit de l'échantillon
    st.sidebar.markdown("<u>Nombre de crédit :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Revenus moyen de l'échantillon
    st.sidebar.markdown("<u>Revenus moyens (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # Montant moyen emprunté de l'échantillon
    st.sidebar.markdown("<u>Montant (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
    # Camembert
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['Solvable', 'Insolvable'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)
    
    
    #######################################
    #             PAGE PRINCIPALE         #
    #######################################
    
    # Solvabilité du client
    
    st.write("Client sélectionné : " + str(selected_id))
    
    #html_temp_4 = """
    #<div style="background-color: white; padding:5px; border-radius:5px">
    #<h2 style="color: black; text-align:center">Solvabilité client</h2>
    #</div> """
    #st.markdown(html_temp_4, unsafe_allow_html=True)   # Affichage de l'identifiant client
    
    proba = proba_load(selected_id)
        
    if proba > 0.3 :
       etat = 'Client à risque'
    else:
       etat = 'Client peu risqué'            
            
    score = round(proba * 100) 
    
    chaine = etat +  ' avec ' + str(score) +'% de risque de défaut de paiement'            
            
    html_temp_3 = """
            <p style="font-size: 15px; font-style: italic">Décision avec un seuil de 30%</p>
            """         

        #affichage de la prédiction
            #st.write('## Prédiction')
    col_1, col_2, col_3, col_4 = st.columns(4)
        
    with col_2:
         gauge_plot(score)

    col_1, col_2, col_3 = st.columns(3)
    with col_2:
         st.markdown(chaine)
         st.markdown(html_temp_3, unsafe_allow_html=True)
# ###################################################################################
            #Feature importance / description                
            #shap.summary_plot(shap_values, features=df_clt, feature_names=df_clt.columns, plot_type='bar')
                         
    st.write('## Explication du résultat') 
    #html_temp_4 = """
    #<div style="background-color: white; padding:10px; border-radius:10px">
    #<h2 style="color: black; text-align:center">Explication du résultat</h2>
    #</div> """
    #st.markdown(html_temp_4, unsafe_allow_html=True)   # Affichage de l'identifiant client    
        
    #Feature importance / description        
    if st.checkbox("Feature Importance "):
        
        shap.initjs()
        
        X = clean_data.iloc[clean_data.index == int(selected_id)].drop(columns='TARGET', axis=1)
        number = st.slider("Nombre de features : ", 0, 20, 5)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        col1, col2 = st.columns(2)
        
        with col1:
             st.subheader("Global")
             fig, ax = plt.subplots(figsize=(15, 15))
             shap.summary_plot(shap_values, X, plot_type ="bar", max_display=number, color_bar=False)
             st.pyplot(fig)

        with col2:
             st.subheader("Local")
             fig, ax = plt.subplots(figsize=(15, 15))
             shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False)
             st.pyplot(fig)


        if st.checkbox("Besoin d'information sur les features ?") :
            list_features = description.Row.to_list()
            feature = st.selectbox('Liste des features', list_features)
            st.table(description.loc[description.Row == feature][:1])
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        


    #st.markdown("<u>Informations client :</u>", unsafe_allow_html=True)
    #st.write(identite_client(initial_data, selected_id))


    st.write('## Détails client')
    # Informations client (détails) : Genre, Age, Statut marital, Nombre d'enfant, Profession …
    html_temp_4 = """
    <div style="background-color: white; padding:10px; border-radius:10px">
    <h2 style="color: black; text-align:center">Informations client</h2>
    </div> """
    #st.markdown(html_temp_4, unsafe_allow_html=True)
    
    #if st.checkbox("Afficher les informations clients ?"):

    general_customer_info = identite_client(initial_data, selected_id)
    sample_customer_info = identite_client(initial_data, selected_id)
        
        ### SEE DATA ###
    row_1_espace_1, row_1_1, row_1_spacer_2 = st.columns((.2, 7.1, .2))
    with row_1_1:
         st.subheader("Profil client")
             
# Profil client label       
    row_2_espace_1, row_2_1, row_2_espace_2, row_2_2, row_2_espace_3, row_2_3, row_2_espace_4, row_2_4, row_2_espace_5, row_2_5, row_2_espace_6   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
    with row_2_1:
         html_temp_4_0 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">GENRE</h5>
             </div> """
         st.markdown(html_temp_4_0, unsafe_allow_html=True)
     
    with row_2_2:
         html_temp_4_1 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">ÂGE</h5>
             </div> """
         st.markdown(html_temp_4_1, unsafe_allow_html=True)
    with row_2_3:
         html_temp_4_2 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">SITUATION FAMILIALE</h5>
             </div> """
         st.markdown(html_temp_4_2, unsafe_allow_html=True)
    with row_2_4:
         html_temp_4_3 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">NOMBRE D'ENFANTS</h5>
             </div> """
         st.markdown(html_temp_4_3, unsafe_allow_html=True)
    with row_2_5:
         html_temp_4_4 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">PROFESSION</h5>
             </div> """
         st.markdown(html_temp_4_4, unsafe_allow_html=True)
             
# Profil client value
    row_2_espace_1_1, row_2_1_1, row_2_espace_2_1, row_2_2_1, row_2_espace_3_1, row_2_3_1, row_2_espace_4_1, row_2_4_1, row_2_espace_5_1, row_2_5_1, row_2_espace_6_1   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
    with row_2_1_1:
         str_gender =  str(general_customer_info["CODE_GENDER"].values[0]) 
         st.markdown(str_gender)
    with row_2_2_1:
         calc_age = int(general_customer_info['DAYS_BIRTH'].abs() / 365.25)
         str_age = str(calc_age)
         st.markdown(str_age)
    with row_2_3_1:
         str_family_status = str(general_customer_info["NAME_FAMILY_STATUS"].values[0]) 
         st.markdown(str_family_status)
    with row_2_4_1:
         str_children_number =  str(general_customer_info["CNT_CHILDREN"].values[0]) 
         st.markdown(str_children_number)
    with row_2_5_1:
         str_profession = str(general_customer_info['OCCUPATION_TYPE'].values[0]) 
         st.markdown(str_profession)


    ### AGE ###
    row2_espace, row2_1, row2_espace_2 = st.columns((.2, 7.1, .2))
    with row2_1:
         st.subheader('Analyse par âge')
        
    row3_espace, row3_1, row3_espace_2, row3_2, row3_espace_3  = st.columns((.2, 2.3, .4, 4.4, .2))
    with row3_1:
         st.markdown("Distribution de l'âge en fonction des clients mais également en fonction de la classe. Dans chacun des graphes, le client sera positionné.")    
         class_selected = st.selectbox ("Quel type de client souhaitez-vous sélectionner ? ", ['Avec défaut de paiement', 'Sans défaut de paiement'], key='age_selectbox')
             
    with row3_2:
         if class_selected == 'Sans défaut de paiement':
            #Age distribution plot
            data_age = load_age_population(clean_data[clean_data.TARGET == 0])
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, edgecolor = 'k', color="#a21251", bins=20)
            ax.axvline(int(general_customer_info["DAYS_BIRTH"].abs() / 365), color="blue", linestyle='--')
            ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
            st.pyplot(fig)
           
         if class_selected == 'Avec défaut de paiement':
            #Age distribution plot
            data_age = load_age_population(initial_data[initial_data.TARGET == 1])
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, edgecolor = 'k', color='#a21251', bins=20)
            ax.axvline(int(general_customer_info["DAYS_BIRTH"].abs() / 365), color="blue", linestyle='--')
            ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
            st.pyplot(fig)
       
    
    ### Revenus client ###
     
    # Revenus client label       
    row_3_espace_1, row_3_1, row_3_espace_2, row_3_2, row_3_espace_3, row_3_3, row_3_espace_4, row_3_4, row_3_espace_5 = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
    
    with row_3_1:
         html_temp_3_0 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">MONTANT REVENUS</h5>
             </div> """
         st.markdown(html_temp_3_0, unsafe_allow_html=True)
     
    with row_3_2:
         html_temp_3_1 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">MONTANT CRÉDIT</h5>
             </div> """
         st.markdown(html_temp_3_1, unsafe_allow_html=True)
    with row_3_3:
         html_temp_3_2 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">MONTANT ANNUITÉS</h5>
             </div> """
         st.markdown(html_temp_3_2, unsafe_allow_html=True)
    with row_3_4:
         html_temp_3_3 = """
             <div style="background-color: white; padding:5px">
             <h5 style="color: black; text-align:left">PRIX DU PRODUIT</h5>
             </div> """
         st.markdown(html_temp_3_3, unsafe_allow_html=True)
      
             
# Revenus client valeur
    row_3_espace_1_1, row_3_1_1, row_3_espace_2_1, row_3_2_1, row_3_espace_3_1, row_3_3_1, row_3_espace_4_1, row_3_4_1, row_3_espace_5_1  = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
    with row_3_1_1:
         str_income =  str(sample_customer_info["AMT_INCOME_TOTAL"].values[0]) 
         st.markdown(str_income)
    with row_3_2_1:
         str_credit = str(sample_customer_info["AMT_CREDIT"].values[0])
         st.markdown(str_credit)
    with row_3_3_1:
         str_annuity = str(sample_customer_info["AMT_ANNUITY"].values[0]) 
         st.markdown(str_annuity)
    with row_3_4_1:
         str_product_price =  str(sample_customer_info["AMT_GOODS_PRICE"].values[0]) 
         st.markdown(str_product_price)

   
        
        ### REVENUS ###
    row4_espace, row4_1, row4_espace_2 = st.columns((.2, 7.1, .2))
    with row4_1:
         st.subheader('Analyse par montant des revenus')
        
    row5_espace, row5_1, row5_espace_2, row5_2, row5_espace_3  = st.columns((.2, 2.3, .4, 4.4, .2))

    with row5_1:
         st.markdown("Distribution du montant des revenus en fonction des clients mais également en fonction de la classe. Dans chacun des graphes, le client sera positionné.")    
         class_selected_2 = st.selectbox ("Quel type de client souhaitez-vous sélectionner ? ", ['Avec défaut de paiement', 'Sans défaut de paiement'], key='income_selectbox')
             
    with row5_2:
         if class_selected_2 == 'Sans défaut de paiement':
         #Income distribution plot
            sel_index_1 = target[target == 0].index.to_list()
            data_income = load_income_population(clean_data[clean_data.index.isin(sel_index_1)])
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color='#a21251', bins=10)
            ax.axvline(int(sample_customer_info["AMT_INCOME_TOTAL"].values[0]), color="blue", linestyle='--')
            ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
            st.pyplot(fig)
           
         if class_selected_2 == 'Avec défaut de paiement':
            #Income distribution plot
            sel_index_2 = target[target == 1].index.to_list()
            data_income = load_income_population(clean_data[clean_data.index.isin(sel_index_2)])
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color='#a21251', bins=10)
            ax.axvline(int(sample_customer_info["AMT_INCOME_TOTAL"].values[0]), color="blue", linestyle='--')
            ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
            st.pyplot(fig) 
                
                
                
                
        ### Analyse bivariée... ###
        
    row6_espace, row6_1, row6_espace_2 = st.columns((.2, 7.1, .2))
    with row6_1:
         st.subheader("Correlation entre l'âge et les revenus")
        
    row7_espace, row7_1, row7_espace_2, row7_2, row7_espace_3  = st.columns((.2, 2.3, .4, 4.4, .2))

    with row7_1:
         st.markdown("Distribution du montant des revenus en fonction des clients mais également en fonction de la classe. Dans chacun des graphes, le client sera positionné.")    
         class_selected_2 = st.selectbox ("Quel type de client souhaitez-vous sélectionner ? ", ['Avec défaut de paiement', 'Sans défaut de paiement'], key='income_age_selectbox')
    
    data_wk = data_wk.reset_index()
    data_wk['AGE'] = (data_wk['DAYS_BIRTH'].abs()/365).round(1)         
    with row7_2:
         if class_selected_2 == 'Sans défaut de paiement':
            #Relationship Age / Income Total interactive plot 
            data_wk['AGE'] = (data_wk['DAYS_BIRTH'].abs()/365).round(1)
            data_wk_f = data_wk[data_wk.TARGET == 0]
            fig, ax = plt.subplots(figsize=(10, 10))
            fig = px.scatter(data_wk_f[data_wk_f.AMT_INCOME_TOTAL < 1000000], x='AGE', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='PRED_PROBA',
                         hover_data=['SK_ID_CURR'])
                         #hover_data=['NAME_FAMILY_STATUS', 'NAME_CONTRACT_TYPE'])

            fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relation Age / Income Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


            fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
            fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
            fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

            st.plotly_chart(fig)
           
         if class_selected_2 == 'Avec défaut de paiement':
            #Relationship Age / Income Total interactive plot 
            data_wk['AGE'] = (data_wk['DAYS_BIRTH'].abs()/365).round(1)
            data_wk_s = data_wk[data_wk.TARGET == 1]
            fig_2, ax = plt.subplots(figsize=(10, 10))
            fig_2 = px.scatter(data_wk_s[data_wk_s.AMT_INCOME_TOTAL < 1000000], x='AGE', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='PRED_PROBA',
                         hover_data=['SK_ID_CURR'])
                         #hover_data=['NAME_FAMILY_STATUS', 'NAME_CONTRACT_TYPE'])

            fig_2.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relation Age / Income Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


            fig_2.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
            fig_2.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
            fig_2.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

            st.plotly_chart(fig_2) 
            
            
            
            
            
            
    row8_espace, row8_1, row8_espace_2 = st.columns((.2, 7.1, .2))
    with row8_1:
         st.subheader("Correlation entre l'âge et les revenus")
        
    row9_espace, row9_1, row9_espace_2, row9_2, row9_espace_3  = st.columns((.2, 2.3, .4, 4.4, .2))

    with row9_1:
         st.markdown("Distribution du montant des revenus en fonction des clients mais également en fonction de la classe. Dans chacun des graphes, le client sera positionné.")    
         class_selected_3 = st.selectbox ("Quel type de client souhaitez-vous sélectionner ? ", ['Avec défaut de paiement', 'Sans défaut de paiement'], key='income_age_selectbox_2')
            
    with row9_2:
         if class_selected_3 == 'Sans défaut de paiement':
            #Relationship Age / Income Total interactive plot 
            data_wk['AGE'] = (data_wk['DAYS_BIRTH'].abs()/365).round(1)
            data_wk_f = data_wk[data_wk.PRED_PROBA < 0.5]
            fig, ax = plt.subplots(figsize=(10, 10))
            fig = px.scatter(data_wk_f[data_wk_f.AMT_INCOME_TOTAL < 1000000], x='AGE', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='TARGET',
                         hover_data=['SK_ID_CURR', 'PRED_PROBA'])
                         #hover_data=['NAME_FAMILY_STATUS', 'NAME_CONTRACT_TYPE'])

            fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relation Age / Income Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


            fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
            fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
            fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

            st.plotly_chart(fig)
           
         if class_selected_3 == 'Avec défaut de paiement':
            #Relationship Age / Income Total interactive plot 
            data_wk['AGE'] = (data_wk['DAYS_BIRTH'].abs()/365).round(1)
            data_wk_s = data_wk[data_wk.PRED_PROBA > 0.5]
            fig_2, ax = plt.subplots(figsize=(10, 10))
            fig_2 = px.scatter(data_wk_s[data_wk_s.AMT_INCOME_TOTAL < 1000000], x='AGE', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='TARGET',
                         hover_data=['SK_ID_CURR', 'PRED_PROBA'])
                         #hover_data=['NAME_FAMILY_STATUS', 'NAME_CONTRACT_TYPE'])

            fig_2.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relation Age / Income Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


            fig_2.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
            fig_2.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
            fig_2.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

            st.plotly_chart(fig_2)
        
    
    #else:
    #    st.markdown("<i>…</i>", unsafe_allow_html=True)"""

 
    #Similar customer files display
    chk_voisins = st.checkbox("Show similar customer files ?")

    if chk_voisins:
        knn = load_knn(clean_data)
        st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
        st.dataframe(load_kmeans(clean_data, selected_id, knn))
        st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        

if __name__ == '__main__':
    main()