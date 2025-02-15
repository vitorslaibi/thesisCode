import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load and preprocess NBA dataset."""
    nba_data = pd.read_csv(file_path)
    categorical_attributes = nba_data.select_dtypes(include=['object']).columns.tolist()
    nba_data = nba_data.drop(['GAME_DATE_EST', 'GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON','TEAM_ID_home','TEAM_ID_away'], axis=1)
    X = nba_data.drop('HOME_TEAM_WINS', axis=1)
    y = nba_data['HOME_TEAM_WINS']
    print(X.columns)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return X, y

def initialize_population(num_features, pop_size):
    """Initialize the population for the genetic algorithm."""
    return np.random.choice([0, 1], size=(pop_size, num_features), replace=True)

def mutate(individual, mutation_rate):
    """Apply mutation to an individual."""
    mutation_points = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_points] = 1 - individual[mutation_points]
    return individual

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    crossover_point = np.random.randint(len(parent1))
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def calculate_fitness(X, y, selected_features):
    """Calculate the fitness of an individual based on selected features."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:, selected_features], y)
    y_pred = model.predict(X[:, selected_features])
    return accuracy_score(y, y_pred)

def apply_idw(X, num_games):
    """Apply Inverse Distance Weighting (IDW) to the features."""
    weights = np.array([1 / (i + 1) for i in range(num_games)])
    idw_features = X[:, -num_games:] * weights
    return np.sum(idw_features, axis=1).reshape(-1, 1)

def run_genetic_algorithm(X_train, y_train, num_features, population_size, generations, mutation_rate):
    """Run the genetic algorithm to select features."""
    population = initialize_population(num_features, population_size)
    for generation in range(generations):
        fitness_scores = [calculate_fitness(X_train, y_train, np.where(individual == 1)[0]) for individual in population]
        selected_indices = np.argsort(fitness_scores)[-population_size:]
        selected_population = population[selected_indices]
        new_population = []
        for i in range(population_size // 2):
            parent1 = selected_population[np.random.choice(population_size)]
            parent2 = selected_population[np.random.choice(population_size)]
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent1, parent2), mutation_rate)
            new_population.extend([child1, child2])
        population = np.array(new_population)
    best_individual = population[np.argmax(fitness_scores)]
    selected_features = np.where(best_individual == 1)[0]
    return selected_features

def evaluate_model(X_train, X_test, y_train, y_test, selected_features):
    """Evaluate the model performance using selected features."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train[:, selected_features], y_train)
    y_pred = model.predict(X_test[:, selected_features])
    print(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main execution
def main():
    file_path = 'data/games.csv'
    X, y = load_data(file_path)
    
    num_games = 5  # Hyperparameter for the number of recent games
    X_idw = apply_idw(X, num_games)
    X = np.hstack((X, X_idw))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    population_size = 10
    generations = 5
    mutation_rate = 0.1
    n_iterations = 10

    feature_selection_counts = np.zeros(X_train_scaled.shape[1])

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}")
        selected_features = run_genetic_algorithm(X_train_scaled, y_train, X_train_scaled.shape[1], population_size, generations, mutation_rate)
        feature_selection_counts[selected_features] += 1

    feature_selection_percentages = (feature_selection_counts / n_iterations) * 100

    print("Feature Selection Percentages:")
    for idx, percentage in enumerate(feature_selection_percentages):
        print(f"Feature {idx}: {percentage:.2f}%")

    selected_features = np.where(feature_selection_percentages > 0)[0]
    accuracy = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, selected_features)
    print(f"Accuracy on Test Set: {accuracy}")

if __name__ == "__main__":
    main()
