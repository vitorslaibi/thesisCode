import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import numpy as np
import random
from collections import Counter
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

def normalize_columns(file_path, vars):
    df = pd.read_csv(file_path)
    df['DRB'] = df['TRB'] - df['ORB']
    columns = df.columns.tolist()
    orbs_index = columns.index('ORB') + 1
    columns.insert(orbs_index, 'DRB')
    columns.remove('DRB')
    df = df[columns]

    for col in vars:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    scaler = MinMaxScaler()
    df[vars] = scaler.fit_transform(df[vars])

    return df

def calculate_aw(l, d):
    denominator = sum(n ** d for n in range(1, l + 1))
    weights = [(l - n + 1) ** d / denominator for n in range(1, l + 1)]
    return weights

def calculate_feature(X0, i, l, d):
    weights = calculate_aw(l, d)
    features = []
    for t in range(l, X0.shape[1]):
        feature_value = sum(weights[n - 1] * X0[i, t - n] for n in range(1, l + 1))
        features.append(feature_value)
    return features

def genetic_algorithm(features_df, target, num_generations=50, population_size=50, 
                     mutation_rate=0.1, tournament_size=3, elite_size=2):
    num_features = features_df.shape[1]
    
    population = []
    for _ in range(population_size):
        chromosome = np.zeros(num_features)
        num_selected = max(int(0.3 * num_features), 1)
        selected_indices = np.random.choice(num_features, num_selected, replace=False)
        chromosome[selected_indices] = 1
        population.append(chromosome)

    def fitness(chromosome):
        selected_features = features_df.iloc[:, chromosome.astype(bool)]
        if selected_features.shape[1] == 0:
            return float('-inf')
        
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(selected_features):
            X_train = selected_features.iloc[train_idx]
            X_val = selected_features.iloc[val_idx]
            y_train = target.iloc[train_idx]
            y_val = target.iloc[val_idx]
            
            model = DecisionTreeRegressor(
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=3,
                max_depth=10
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred)) 
            cv_scores.append(-rmse)
            
        feature_penalty = -0.01 * np.sum(chromosome)
        return np.mean(cv_scores) + feature_penalty

    def tournament_selection(population, fitness_scores):
        selected = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in selected]
        winner_idx = selected[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def adaptive_crossover(parent1, parent2, generation, max_generations):
        crossover_rate = 0.7 + 0.2 * (generation / max_generations)
        if random.random() < crossover_rate:
            points = sorted(random.sample(range(num_features), 2))
            child1 = np.concatenate([parent1[:points[0]], 
                                   parent2[points[0]:points[1]], 
                                   parent1[points[1]:]])
            child2 = np.concatenate([parent2[:points[0]], 
                                   parent1[points[0]:points[1]], 
                                   parent2[points[1]:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return child1, child2

    def adaptive_mutation(chromosome, generation, max_generations):
        current_mutation_rate = mutation_rate * (1 - 0.5 * (generation / max_generations))
        for i in range(num_features):
            if random.random() < current_mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        
        if np.sum(chromosome) == 0:
            chromosome[random.randint(0, num_features-1)] = 1

    best_fitness = float('-inf')
    best_solution = None
    generations_without_improvement = 0
    
    for generation in range(num_generations):
        fitness_scores = [fitness(chromosome) for chromosome in population]
        
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best_solution = population[current_best_idx].copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
            
        if generations_without_improvement >= 10:
            break
            
        sorted_pairs = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
        next_generation = [pair[1].copy() for pair in sorted_pairs[:elite_size]]
        
        while len(next_generation) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            child1, child2 = adaptive_crossover(parent1, parent2, generation, num_generations)
            
            adaptive_mutation(child1, generation, num_generations)
            adaptive_mutation(child2, generation, num_generations)
            
            next_generation.extend([child1, child2])
        
        population = next_generation[:population_size]
    
    return best_solution

def run_multiple_ga(features_df, target, n_runs=10, selection_threshold=0.25):
    """
    Run genetic algorithm multiple times and select features that appear frequently
    """
    feature_names = features_df.columns
    feature_selections = []
    
    print(f"\nRunning Genetic Algorithm {n_runs} times")
    for i in range(n_runs):
        print(f"\nGA Run {i+1}/{n_runs}")
        best_features_mask = genetic_algorithm(features_df, target)
        selected_features = feature_names[best_features_mask.astype(bool)].tolist()
        feature_selections.extend(selected_features)
    
    # Count feature occurrences
    feature_counts = Counter(feature_selections)
    
    # Calculate selection frequency for each feature
    feature_frequencies = {feature: count/n_runs for feature, count in feature_counts.items()}
    
    # Select features that appear more than threshold times
    final_features = [feature for feature, freq in feature_frequencies.items() 
                     if freq >= selection_threshold]
    
    print("\nFeature Selection Statistics:")
    print("Feature : Selection Frequency")
    for feature, freq in sorted(feature_frequencies.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {freq*100:.1f}%")
    
    print(f"\nFinal selected features (appeared in >{selection_threshold*100}% of runs):")
    print(final_features)
    
    return final_features

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models on the same data
    """
    models = {
        'Decision Tree': DecisionTreeRegressor(
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=3,
            max_depth=10
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=3,
            max_depth=10
        ),
        'Stochastic GB': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=3,
            max_depth=5
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            max_depth=5
        ),
        'ELM (MLP)': MLPRegressor(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            random_state=42,
            max_iter=1000
        )
    }
    
    results = {}
    feature_importance_dict = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions)/2)
        r2 = r2_score(y_test, predictions)
        
        results[name] = {
            'RMSE': rmse,
            'R2': r2
        }
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance_dict[name] = model.feature_importances_
    
    return results, feature_importance_dict

def main():
    # Number of GA runs
    n_runs = 10
    selection_threshold = 0.3
    
    input_file = 'BasketballData/nba_game_logs_2018_2019.csv'
    output_file = 'BasketballData/normalized_nba_game_logs_2018_2019.csv'
    vars = ['FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

    print("Loading and normalizing data...")
    normalized_df = normalize_columns(input_file, vars)
    normalized_df.to_csv(output_file, index=False)
    print(f"Normalized data saved to {output_file}")

    print("Calculating features...")
    X0 = normalized_df.values.T
    l = 3 # Game lag
    d = 3 # Weighting control

    features_dict = {}
    for var in vars:
        i = normalized_df.columns.get_loc(var)
        features = calculate_feature(X0, i, l, d)
        features_dict[var] = features

    features_cart = pd.DataFrame(features_dict)
    target_cart = normalized_df['Tm'].iloc[l:].reset_index(drop=True)

    # Run multiple GA iterations and get consistently selected features
    final_features = run_multiple_ga(features_cart, target_cart, n_runs, selection_threshold)

    # Use only consistently selected features for prediction
    print("\nTraining models with consistently selected features...")
    selected_features_cart = features_cart[final_features]
    actual_values = normalized_df['Tm'].values

    # Create train/test split for final evaluation
    split_idx = int(len(selected_features_cart) * 0.8)
    X_train = selected_features_cart.iloc[:split_idx]
    y_train = actual_values[l:l+split_idx]
    X_test = selected_features_cart.iloc[split_idx:]
    y_test = actual_values[l+split_idx:l+len(selected_features_cart)]

    # Train and evaluate all models
    results, feature_importance_dict = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Print results
    print("\nModel Performance Comparison:")
    print("-" * 60)
    print(f"{'Model':<15} {'RMSE':>12} {'R²':>12}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['RMSE']:>12.4f} {metrics['R2']:>12.4f}")
    print("-" * 60)

if __name__ == '__main__':
    main()
