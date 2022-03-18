import streamlit as st
import streamlit.components.v1 as components
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
from PIL import Image

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
                 {'range': [0, 40], 'color': "#d4e6fa"},
                 {'range': [40, 60], 'color': "#7fb6f1"},
                 {'range': [60, 100], 'color': "#2b86e9"} ],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 40}}))

        st.plotly_chart(fig)
        

    def identite_client(data, customer_id):
        data_client = data[data.index == int(customer_id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"].abs() /365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = sample[sample['AMT_INCOME_TOTAL'] < 300000]['AMT_INCOME_TOTAL']
        return df_income


    def proba_load(customer_id):
        
        API_url = "https://get-scoring-credit.herokuapp.com/get_score_credit/" + str(customer_id)
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        pred_proba = API_data['score']
        
        return pred_proba  

    def hist_plot(data_var, client_value, title, x_label):
        
        rc = {'figure.figsize':(5,5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
        
        plt.rcParams.update(rc)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_var, edgecolor = 'k', color="#a21251", bins=20)
        ax.axvline(client_value, color="blue", linestyle='--')
        ax.set(title=title, xlabel=x_label, ylabel='')
        st.pyplot(fig)
        
    def shap_plot(shap_values, data):
        
        fig, ax = plt.subplots(figsize=(15, 15))
        shap.summary_plot(shap_values, data, plot_type ="bar", max_display=number, color_bar=True)
        st.pyplot(fig)

    def plot_2_variables(data, x_var, y_var, title, color_var, hover_data_var, client_value_x, client_value_y):
                   
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data, x=x_var, y=y_var, size=y_var, color=color_var, hover_data=hover_data_var)
        fig.update_layout({'plot_bgcolor':'#0e1117'}, 
                          title={'text':title, 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))

        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#5b5b5b',
                         title=x_var, title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#5b5b5b',
                         title=y_var, title_font=dict(size=18, family='Verdana'))
        # Ligne client horizontal                 
        fig.add_shape(type="line", x0=0, y0=client_value_y, x1=max(data[x_var]), y1=client_value_y, line=dict(color="white", width=3, dash="dot" ))
        # Ligne client vertical
        fig.add_shape(type="line", x0=client_value_x, y0=0, x1=client_value_x, y1=max(data[y_var]), line=dict(color="white", width=3, dash="dot" ))        

        st.plotly_chart(fig)  

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)        
        
   
        
    # Chargement des données
    initial_data, clean_data, target, description = load_data()
    model = load_model()
    customer_id = clean_data.index.values
    data_wk = clean_data.copy()
    data_wk['SCORE'] = model.predict_proba(data_wk.drop(columns='TARGET', axis=1))[:,1] * 100
    data_wk['AGE'] = (data_wk['DAYS_BIRTH'].abs()/365).round(1)  
    image = Image.open('image/pret_a_depenser_logo.png')
    


    #######################################
    #               SIDEBAR               #
    #######################################

    # TITRE
    html_temp = """
    <div style="background-color: #0e1117; padding:10px; border-radius:10px">
    <h1 style="color: #bcbcbc; text-align:left">PRÊT À DÉPENSER - Tableau de bord</h1>
    <p style="color: #bcbcbc;font-size: 20px; font-style: italic; text-align:left">Aide à la décision d'octroi de crédit</p></div>
    """
    col_1,col_2,col_3 = st.columns((1, 5, 2))
    with col_1:
         st.image(image)
    with col_2:
         st.markdown(html_temp, unsafe_allow_html=True)
    with col_3:
         st.text("streamlit app par adondo")


    # Selection des identifiants clients
    st.sidebar.header("** Informations générales **")

    #Loading selectbox
    selected_id = st.sidebar.selectbox("Client ID", customer_id)

    
    sidebar_selection = st.sidebar.radio(
        ' ',
        ['Score de solvabilité client', 'Comparaison des clients'],
        )


    if sidebar_selection == 'Score de solvabilité client':
 
    
       #######################################
       #             PAGE PRINCIPALE         #
       #######################################
    
       # Chargement et traitement des données de prédiction

       proba = proba_load(selected_id)
        
       if proba >= 0.4 :
          etat = 'Client à risque'
          if proba >= 0.6 :
             etat = 'Client très risqué'
       else:
          etat = 'Client peu risqué'            
            
       score = round(proba * 100) 
    
       chaine = etat +  ' avec ' + str(score) +'% de risque de défaut de paiement'            
            
       html_temp_3 = """
            <p style="font-size: 15px; font-style: italic">Décision avec un seuil de 40%</p>
            """         

       #affichage de la prédiction
       col_1, col_2, col_3, col_4 = st.columns(4)
        
       with col_2:
            gauge_plot(score)

       col_1, col_2, col_3 = st.columns(3)
       with col_2:
            st.markdown(chaine)
            st.markdown(html_temp_3, unsafe_allow_html=True)

           
    # Affichage des features importance       
       st.write('## Explication du résultat') 
    
    #Feature importance / description        
       if st.checkbox("Feature Importance "):
          
          # Chargement des données propres au client sélectionné
          X_selected = clean_data.iloc[clean_data.index == int(selected_id)].drop(columns='TARGET', axis=1)
          X_all = clean_data.drop(columns='TARGET', axis=1)
          number = st.slider("Nombre de features : ", 0, 20, 5)
        
          explainer_tree = shap.TreeExplainer(model)
          shap_values_tree = explainer_tree.shap_values(X_selected)
          
          explainer = shap.Explainer(model)
          shap_values = explainer.shap_values(X_all)
          
          shap.initjs()
           
          st_shap(shap.force_plot(explainer_tree.expected_value[0], shap_values_tree[0], X_selected))
          
          
          col1, col2 = st.columns(2)
        
          with col1:
               # Features importance globales
               st.subheader("Global")
               shap_plot(shap_values, X_all)

          with col2:
               # Features importance locales
               st.subheader("Local")
               shap_plot(shap_values[1], X_selected)

          # Informations sur les variables
          if st.checkbox("Besoin d'information sur les features ?") :
              list_features = description.Row.to_list()
              feature = st.selectbox('Liste des features', list_features)
              st.table(description.loc[description.Row == feature][:1])
        
       else:
          st.markdown("<i>…</i>", unsafe_allow_html=True)
        


    # Comparaison des clients

    if sidebar_selection == 'Comparaison des clients':
       st.write('## Détails client')
       # Chargement des informations du client
       general_customer_info = identite_client(initial_data, selected_id)
       sample_customer_info = identite_client(initial_data, selected_id)
       customer_age_value = int(general_customer_info["DAYS_BIRTH"].abs() / 365)
       customer_income_value = int(sample_customer_info["AMT_INCOME_TOTAL"].values[0])
       
       col_1, col_2, col_3, col_4, col_5 = st.columns((.2, 2.3, .4, 4.4, .2))
        
       with col_2:
            # Choix des features à afficher (distribution de la donnée)
            var_selected = st.selectbox ("Quelle variable souhaitez-vous analyser ? ", ['Âge', 'Revenus'], key='var_selectbox')

       
       if var_selected == 'Âge':
        
       
          row_1_espace_1, row_1_1, row_1_spacer_2 = st.columns((.2, 7.1, .2))
          with row_1_1:
               st.subheader("Informations client")
             
         # Affichage des données d'informations client     
          row_2_espace_1, row_2_1, row_2_espace_2, row_2_2, row_2_espace_3, row_2_3, row_2_espace_4, row_2_4, row_2_espace_5, row_2_5, row_2_espace_6   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
          with row_2_1:
               html_temp_4_0 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">GENRE</h5>
               </div> """
               st.markdown(html_temp_4_0, unsafe_allow_html=True)
     
          with row_2_2:
               html_temp_4_1 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">ÂGE</h5>
               </div> """
               st.markdown(html_temp_4_1, unsafe_allow_html=True)
          with row_2_3:
               html_temp_4_2 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">SITUATION FAMILIALE</h5>
               </div> """
               st.markdown(html_temp_4_2, unsafe_allow_html=True)
          with row_2_4:
               html_temp_4_3 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">NOMBRE D'ENFANTS</h5>
               </div> """
               st.markdown(html_temp_4_3, unsafe_allow_html=True)
          with row_2_5:
               html_temp_4_4 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">PROFESSION</h5>
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
          row_3_espace, row_3_1, row_3_espace_2 = st.columns((.2, 7.1, .2))
          with row_3_1:
               st.subheader('Analyse par âge')
        
          row_4_espace, row_4_1, row_4_espace_2, row_4_2, row_4_espace_3  = st.columns((.2, 2.3, .4, 4.4, .2))
          with row_4_1:
               st.markdown("Distribution de l'âge en fonction du risque du client. Dans chacun des graphes, le client est positionné.")    
               class_selected = st.selectbox ("Quelle population souhaitez-vous afficher ? ", ['Avec défaut de paiement', 'Sans défaut de paiement'], key='age_selectbox')
             
          with row_4_2:
               if class_selected == 'Sans défaut de paiement':
            #Age distribution plot
                  data_age = load_age_population(initial_data[initial_data.TARGET == 0])
                  hist_plot(data_age, customer_age_value , 'Customer Age', 'Age (year)')
           
               if class_selected == 'Avec défaut de paiement':
            #Age distribution plot
                  data_age = load_age_population(initial_data[initial_data.TARGET == 1])
                  hist_plot(data_age, customer_age_value , 'Customer Age', 'Age (year)')
       
    
    ### Revenus client ###
       if var_selected == 'Revenus':
          row_1_espace_1, row_1_1, row_1_espace_2 = st.columns((.2, 7.1, .2))
          with row_1_1:
               st.subheader("Informations client")
    # Revenus client label       
          row_2_espace_1, row_2_1, row_2_espace_2, row_2_2, row_2_espace_3, row_2_3, row_2_espace_4, row_2_4, row_2_espace_5 = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
    
          with row_2_1:
               html_temp_3_0 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">MONTANT REVENUS</h5>
               </div> """
               st.markdown(html_temp_3_0, unsafe_allow_html=True)
     
          with row_2_2:
               html_temp_3_1 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">MONTANT CRÉDIT</h5>
               </div> """
               st.markdown(html_temp_3_1, unsafe_allow_html=True)
          with row_2_3:
               html_temp_3_2 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">MONTANT ANNUITÉS</h5>
               </div> """
               st.markdown(html_temp_3_2, unsafe_allow_html=True)
          with row_2_4:
               html_temp_3_3 = """
               <div style="background-color: #0e1117; padding:5px">
               <h5 style="color: white; text-align:left">PRIX DU PRODUIT</h5>
               </div> """
               st.markdown(html_temp_3_3, unsafe_allow_html=True)
      
             
# Revenus client valeur
          row_2_espace_1_1, row_2_1_1, row_2_espace_2_1, row_2_2_1, row_2_espace_3_1, row_2_3_1, row_2_espace_4_1, row_2_4_1, row_2_espace_5_1  = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
          with row_2_1_1:
               str_income =  str(sample_customer_info["AMT_INCOME_TOTAL"].values[0]) 
               st.markdown(str_income)
          with row_2_2_1:
               str_credit = str(sample_customer_info["AMT_CREDIT"].values[0])
               st.markdown(str_credit)
          with row_2_3_1:
               str_annuity = str(sample_customer_info["AMT_ANNUITY"].values[0]) 
               st.markdown(str_annuity)
          with row_2_4_1:
               str_product_price =  str(sample_customer_info["AMT_GOODS_PRICE"].values[0]) 
               st.markdown(str_product_price)

   
        
        ### REVENUS ###
          row_3_espace, row_3_1, row_3_espace_2 = st.columns((.2, 7.1, .2))
          with row_3_1:
               st.subheader('Analyse par montant des revenus')
        
          row_4_espace, row_4_1, row_4_espace_2, row_4_2, row_4_espace_3  = st.columns((.2, 2.3, .4, 4.4, .2))
 
          with row_4_1:
               st.markdown("Distribution du montant des revenus en fonction du risque du client. Dans chacun des graphes, le client sera positionné.")    
               class_selected_2 = st.selectbox ("Quelle population souhaitez-vous afficher ? ", ['Avec défaut de paiement', 'Sans défaut de paiement'], key='income_selectbox')
             
          with row_4_2:
               if class_selected_2 == 'Sans défaut de paiement':
         #Income distribution plot
                  data_income = load_income_population(initial_data[initial_data.TARGET == 0])
                  hist_plot(data_income, customer_income_value , 'Customer Income', 'Income (USD)')
            
               if class_selected_2 == 'Avec défaut de paiement':
            #Income distribution plot
                  data_income = load_income_population(initial_data[initial_data.TARGET == 1])
                  hist_plot(data_income, customer_income_value , 'Customer Income', 'Income (USD)') 
                
                
                
                
        ### Analyse bivariée... ###
        
       row_5_espace, row_5_1, row_5_espace_2 = st.columns((.2, 7.1, .2))
       with row_5_1:
            st.subheader("Correlation entre l'âge et les revenus")
         
       row_6_espace, row_6_1, row_6_espace_2, row_6_2, row_6_espace_3  = st.columns((.2, 2.3, .4, 4.4, .2))

       with row_6_1:
            st.markdown("Graphique d'analyse des revenus en fonction de l'âge du client. Un graphique selon le risque client. L'épaisseur des points augmente avec les revenus et leur couleur change en fonction du score client. Dans chacun des graphes, le client sera positionné.")    
            class_selected_2 = st.selectbox ("Quel type de client souhaitez-vous sélectionner ? ", ['Avec défaut de paiement', 'Sans défaut de paiement'], key='income_age_selectbox')
    
       data_wk = data_wk.reset_index()
              
       with row_6_2:
            if class_selected_2 == 'Sans défaut de paiement':
            # Correlation Age / Income Total  
               #data_wk['AGE'] = (data_wk['DAYS_BIRTH'].abs()/365).round(1)
               data_wk_f = data_wk.loc[(data_wk.TARGET == 0) & (data_wk.AMT_INCOME_TOTAL < 1000000),:]
               plot_2_variables(data_wk_f, 'AGE', 'AMT_INCOME_TOTAL', 'Relation Age / Income Total', 'SCORE', ['SK_ID_CURR'], customer_age_value, customer_income_value)
           
            if class_selected_2 == 'Avec défaut de paiement':
            #Correlation Age / Income Total  
               data_wk_s = data_wk.loc[(data_wk.TARGET == 1) & (data_wk.AMT_INCOME_TOTAL < 1000000),:]
               plot_2_variables(data_wk_s, 'AGE', 'AMT_INCOME_TOTAL', 'Relation Age / Income Total', 'SCORE', ['SK_ID_CURR'], customer_age_value, customer_income_value)

     
               
        
    

if __name__ == '__main__':
    main()