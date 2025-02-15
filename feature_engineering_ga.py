
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

# Feature Engineering Functions (from comp.py)
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

# Genetic Algorithm Functions (from genetic_algorithm_idw.py)
def load_data(file_path, selected_columns=None):
    df = pd.read_csv(file_path)
    if selected_columns:
        df = df[selected_columns]
    
    X = df.drop('HOME_TEAM_WINS', axis=1)
    y = df['HOME_TEAM_WINS']

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y

def initialize_population(num_features, pop_size):
    return np.random.choice([0, 1], size=(pop_size, num_features))

def fitness(individual, X, y):
    selected_features = [i for i in range(len(individual)) if individual[i] == 1]
    if len(selected_features) == 0:
        return 0
    X_selected = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    return accuracy_score(y_test, predictions)

def selection(population, fitnesses):
    selected = np.random.choice(len(population), size=2, p=fitnesses/np.sum(fitnesses), replace=False)
    return population[selected[0]], population[selected[1]]

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

def mutation(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm(X, y, num_generations=50, pop_size=20, mutation_rate=0.01):
    num_features = X.shape[1]
    population = initialize_population(num_features, pop_size)
    
    for generation in range(num_generations):
        fitnesses = np.array([fitness(ind, X, y) for ind in population])
        
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutation(child1, mutation_rate), mutation(child2, mutation_rate)])
        
        population = new_population
        
        best_fitness = np.max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_individual, best_fitness

# Main function to combine feature engineering and genetic algorithm
def main():
    input_file = 'nba_game_logs_2018_2019.csv'
    output_file = 'engineered_nba_game_logs_2018_2019.csv'
    vars = ['FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

    # Step 1: Feature Engineering
    normalized_df = normalize_columns(input_file, vars)
    X0 = normalized_df.values.T
    l, d = 6, 1

    features_dict = {}
    for var in vars:
        i = normalized_df.columns.get_loc(var)
        features = calculate_feature(X0, i, l, d)
        features_dict[var] = features
    
    features_cart = pd.DataFrame(features_dict)
    features_cart.to_csv(output_file, index=False)
    print(f"Engineered features saved to {output_file}")

    # Step 2: Genetic Algorithm for Feature Selection
    X, y = load_data(output_file)
    best_individual, best_fitness = genetic_algorithm(X, y)
    
    print(f"Best individual (selected features): {best_individual}")
    print(f"Best fitness: {best_fitness}")

if __name__ == '__main__':
    main()
