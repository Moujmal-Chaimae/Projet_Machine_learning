"""
Script pour générer des données de test réalistes pour le projet.
Ces données simulent des réservations d'hôtel avec des caractéristiques réalistes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_hotel_bookings_data(n_samples=10000, random_state=42):
    """
    Génère des données de réservations d'hôtel réalistes.
    
    Args:
        n_samples: Nombre de réservations à générer
        random_state: Seed pour la reproductibilité
    
    Returns:
        DataFrame avec les données générées
    """
    np.random.seed(random_state)
    
    print(f"Génération de {n_samples:,} réservations d'hôtel...")
    
    # Définir les valeurs possibles pour les variables catégorielles
    hotels = ['Resort Hotel', 'City Hotel']
    meals = ['BB', 'HB', 'FB', 'SC', 'Undefined']
    countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'NLD', 'USA']
    market_segments = ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups', 'Complementary', 'Aviation']
    distribution_channels = ['TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined']
    reserved_room_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']
    assigned_room_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'I', 'K']
    deposit_types = ['No Deposit', 'Non Refund', 'Refundable']
    customer_types = ['Transient', 'Contract', 'Transient-Party', 'Group']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    # Générer les données
    data = {
        # Informations de base
        'hotel': np.random.choice(hotels, n_samples, p=[0.4, 0.6]),
        
        # Dates
        'arrival_date_year': np.random.choice([2015, 2016, 2017], n_samples, p=[0.3, 0.35, 0.35]),
        'arrival_date_month': np.random.choice(months, n_samples),
        'arrival_date_week_number': np.random.randint(1, 53, n_samples),
        'arrival_date_day_of_month': np.random.randint(1, 32, n_samples),
        
        # Durée du séjour
        'stays_in_weekend_nights': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'stays_in_week_nights': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], n_samples, 
                                                  p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02]),
        
        # Invités
        'adults': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.6, 0.15, 0.05]),
        'children': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.1, 0.05]),
        'babies': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        
        # Informations de réservation
        'meal': np.random.choice(meals, n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
        'country': np.random.choice(countries, n_samples),
        'market_segment': np.random.choice(market_segments, n_samples, 
                                          p=[0.45, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02]),
        'distribution_channel': np.random.choice(distribution_channels, n_samples,
                                                p=[0.8, 0.1, 0.05, 0.03, 0.02]),
        
        # Historique client
        'is_repeated_guest': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
        'previous_cancellations': np.random.choice([0, 1, 2, 3], n_samples, p=[0.9, 0.06, 0.03, 0.01]),
        'previous_bookings_not_canceled': np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                                           p=[0.85, 0.08, 0.04, 0.02, 0.005, 0.005]),
        
        # Chambres
        'reserved_room_type': np.random.choice(reserved_room_types, n_samples),
        'assigned_room_type': np.random.choice(assigned_room_types, n_samples),
        
        # Modifications et demandes
        'booking_changes': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.7, 0.2, 0.06, 0.03, 0.01]),
        'deposit_type': np.random.choice(deposit_types, n_samples, p=[0.88, 0.1, 0.02]),
        'days_in_waiting_list': np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                                 p=[0.95, 0.02, 0.01, 0.01, 0.005, 0.005]),
        'customer_type': np.random.choice(customer_types, n_samples, p=[0.75, 0.15, 0.08, 0.02]),
        
        # Tarif et services
        'adr': np.random.gamma(shape=2, scale=40, size=n_samples),  # Average Daily Rate
        'required_car_parking_spaces': np.random.choice([0, 1, 2], n_samples, p=[0.92, 0.07, 0.01]),
        'total_of_special_requests': np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                                      p=[0.6, 0.25, 0.1, 0.03, 0.015, 0.005]),
    }
    
    # Créer le DataFrame
    df = pd.DataFrame(data)
    
    # Ajouter lead_time (temps entre réservation et arrivée)
    # Plus le lead_time est long, plus la probabilité d'annulation est élevée
    df['lead_time'] = np.random.gamma(shape=2, scale=50, size=n_samples).astype(int)
    df['lead_time'] = df['lead_time'].clip(0, 737)  # Max observé dans les vraies données
    
    # Générer la variable cible (is_canceled) de manière réaliste
    # Facteurs influençant l'annulation:
    # - Lead time élevé -> plus d'annulations
    # - Deposit type "No Deposit" -> plus d'annulations
    # - Previous cancellations -> plus d'annulations
    # - Market segment "Online TA" -> plus d'annulations
    
    cancellation_prob = np.zeros(n_samples)
    
    # Base probability
    cancellation_prob += 0.2
    
    # Lead time effect (plus c'est long, plus de risque)
    cancellation_prob += (df['lead_time'] / 737) * 0.3
    
    # Deposit type effect
    cancellation_prob += (df['deposit_type'] == 'No Deposit') * 0.2
    cancellation_prob -= (df['deposit_type'] == 'Non Refund') * 0.3
    
    # Previous cancellations effect
    cancellation_prob += df['previous_cancellations'] * 0.15
    
    # Market segment effect
    cancellation_prob += (df['market_segment'] == 'Online TA') * 0.1
    cancellation_prob -= (df['market_segment'] == 'Direct') * 0.05
    
    # Special requests effect (moins de demandes spéciales -> plus d'annulations)
    cancellation_prob -= df['total_of_special_requests'] * 0.05
    
    # Repeated guest effect
    cancellation_prob -= df['is_repeated_guest'] * 0.15
    
    # Clip probabilities
    cancellation_prob = np.clip(cancellation_prob, 0, 0.9)
    
    # Generate cancellations
    df['is_canceled'] = (np.random.random(n_samples) < cancellation_prob).astype(int)
    
    # Arrondir ADR à 2 décimales
    df['adr'] = df['adr'].round(2)
    
    # S'assurer que ADR est positif
    df['adr'] = df['adr'].clip(0, None)
    
    return df

def main():
    """Fonction principale."""
    print("="*80)
    print("GÉNÉRATION DE DONNÉES DE TEST")
    print("="*80)
    print()
    
    # Créer le répertoire data/raw s'il n'existe pas
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Répertoire créé: {data_dir}")
    
    # Générer les données
    df = generate_hotel_bookings_data(n_samples=10000, random_state=42)
    
    # Afficher des statistiques
    print()
    print("="*80)
    print("STATISTIQUES DES DONNÉES GÉNÉRÉES")
    print("="*80)
    print(f"\nNombre total de réservations: {len(df):,}")
    print(f"Nombre de colonnes: {len(df.columns)}")
    print(f"\nTaux d'annulation: {df['is_canceled'].mean():.2%}")
    print(f"  - Annulées: {df['is_canceled'].sum():,}")
    print(f"  - Non annulées: {(~df['is_canceled'].astype(bool)).sum():,}")
    
    print(f"\nRépartition par type d'hôtel:")
    print(df['hotel'].value_counts())
    
    print(f"\nStatistiques du lead time:")
    print(f"  - Moyenne: {df['lead_time'].mean():.1f} jours")
    print(f"  - Médiane: {df['lead_time'].median():.1f} jours")
    print(f"  - Min: {df['lead_time'].min()} jours")
    print(f"  - Max: {df['lead_time'].max()} jours")
    
    print(f"\nStatistiques de l'ADR (tarif moyen):")
    print(f"  - Moyenne: {df['adr'].mean():.2f}€")
    print(f"  - Médiane: {df['adr'].median():.2f}€")
    print(f"  - Min: {df['adr'].min():.2f}€")
    print(f"  - Max: {df['adr'].max():.2f}€")
    
    # Sauvegarder les données
    output_path = data_dir / 'hotel_bookings.csv'
    df.to_csv(output_path, index=False)
    
    print()
    print("="*80)
    print("SAUVEGARDE")
    print("="*80)
    print(f"\n✓ Données sauvegardées: {output_path}")
    print(f"  Taille du fichier: {output_path.stat().st_size / 1024:.2f} KB")
    
    # Afficher un aperçu
    print()
    print("="*80)
    print("APERÇU DES DONNÉES (5 premières lignes)")
    print("="*80)
    print()
    print(df.head().to_string())
    
    print()
    print("="*80)
    print("✓ GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
    print("="*80)
    print()
    print("Prochaines étapes:")
    print("  1. Exécuter le pipeline: python run_pipeline.py")
    print("  2. Ou lancer l'application: streamlit run app/streamlit_app.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
