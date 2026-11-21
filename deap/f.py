# Clasificador AdaBoost por defecto
default_clf = AdaBoostClassifier(random_state=RANDOM_SEED)

# Validacion cruzada
cv_results = model_selection.cross_val_score(
    default_clf,
    X,
    y,
    cv=kfold,
    scoring="accuracy",
)
default_score = cv_results.mean()
print(
    f"Precision media con valores por defecto: "
    f"{default_score:.4f} ({default_score * 100:.2f}%)"
)

# ///////

# Malla de parametros (200 combinaciones)
param_grid = {
    "n_estimators": [2 - 5],
    "learning_rate": np.logspace(-2, 0, num=10, base=10),
    "algorithm": ["SAMME", "SAMME.R"],
}

# 1. GridSearchCV Tradicional (Exhaustivo)
start_time = time.time()
grid_search = GridSearchCV(
    estimator=AdaBoostClassifier(random_state=RANDOM_SEED),
    param_grid=param_grid,
    cv=kfold,
    scoring="accuracy",
)
grid_search.fit(X, y)
grid_time = time.time() - start_time

print("--- 1. GridSearchCV Tradicional ---")
print(f"Mejores parametros: {grid_search.best_params_}")
print(f"Precision: {grid_search.best_score_:.4f}")
print(f"Tiempo: {grid_time:.2f} segundos\n")

# 2. EvolutionaryAlgorithmSearchCV (sklearn-deap)
start_time = time.time()
genetic_grid = EvolutionaryAlgorithmSearchCV(
    estimator=AdaBoostClassifier(random_state=RANDOM_SEED),
    params=param_grid,
    cv=kfold,
    scoring="accuracy",
    verbose=True,
    population_size=20,
    gene_mutation_prob=0.30,
    tournament_size=2,
    generations_number=5,
)
genetic_grid.fit(X, y)
genetic_grid_time = time.time() - start_time

print("--- 2. EvolutionaryAlgorithmSearchCV (Genetico) ---")
print(f"Mejores parametros: {genetic_grid.best_params_}")
print(f"Precision: {genetic_grid.best_score_:.4f}")
print(f"Tiempo: {genetic_grid_time:.2f} segundos")


# ////////

# Definicion de limites [n_estimators, learning_rate, algorithm]
BOUNDS_LOW = [1.0, 0.01, 0.0]
BOUNDS_HIGH = [100.0, 1.00, 1.0]
NUM_OF_PARAMS = len(BOUNDS_LOW)


class HyperparameterTuningGenetic:
    def __init__(self, X, y, kfold, seed=42):
        self.X = X
        self.y = y
        self.kfold = kfold
        self.seed = seed

    def convert_params(self, params):
        n_estimators = int(round(params))
        learning_rate = float(params[6])
        algorithm = ["SAMME", "SAMME.R"][int(round(params[7]))]
        return n_estimators, learning_rate, algorithm

    def get_accuracy(self, params):
        n_estimators, learning_rate, algorithm = self.convert_params(params)
        clf = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=self.seed,
        )
        cv_results = model_selection.cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.kfold,
            scoring="accuracy",
        )
        return cv_results.mean()


evaluator = HyperparameterTuningGenetic(X, y, kfold, RANDOM_SEED)

# Configuracion de DEAP
if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
if "Individual" in creator.__dict__:
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
for i in range(NUM_OF_PARAMS):
    toolbox.register(
        f"hyperparameter_{i}",
        random.uniform,
        BOUNDS_LOW[i],
        BOUNDS_HIGH[i],
    )

hyperparameters = tuple(
    getattr(toolbox, f"hyperparameter_{i}")
    for i in range(NUM_OF_PARAMS)
)
toolbox.register(
    "individualCreator",
    tools.initCycle,
    creator.Individual,
    hyperparameters,
    n=1,
)
toolbox.register(
    "populationCreator",
    tools.initRepeat,
    list,
    toolbox.individualCreator,
)


def classification_accuracy(individual):
    return (evaluator.get_accuracy(individual),)


toolbox.register("evaluate", classification_accuracy)

# Operadores de seleccion, cruce y mutacion
CROWDING_FACTOR = 20.0
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=CROWDING_FACTOR,
)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=CROWDING_FACTOR,
    indpb=1.0 / NUM_OF_PARAMS,
)

# Ejecucion evolutiva con elitismo
population = toolbox.populationCreator(n=20)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

P_CROSSOVER, P_MUTATION, MAX_GENERATIONS = 0.9, 0.1, 5

# Evaluar poblacion inicial
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

hof.update(population)
print("Gen\tN_Evals\tMax_Accuracy\tAvg_Accuracy")
record = stats.compile(population)
print(f"0\t20\t{record['max']:.5f}\t{record['avg']:.5f}")

for gen in range(1, MAX_GENERATIONS + 1):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < P_MUTATION:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Elitismo
    offspring.sort(key=lambda ind: ind.fitness.values, reverse=True)
    offspring[-1] = toolbox.clone(hof)
    population = offspring
    hof.update(population)

    record = stats.compile(population)
    print(
        f"{gen}\t{len(invalid_ind)}\t"
        f"{record['max']:.5f}\t{record['avg']:.5f}"
    )

best_ind = hof
best_n, best_lr, best_alg = evaluator.convert_params(best_ind)
print("\n--- Resultado de la Optimizacion Genetica Directa ---")
print(
    f"Hiperparametros -> n_estimators: {best_n}, "
    f"learning_rate: {best_lr:.4f}, algorithm: '{best_alg}'"
)
print(
    f"Precision Maxima alcanzada: {best_ind.fitness.values:.5f} "
    f"({best_ind.fitness.values * 100:.2f}%)"
)

# ///////

# Grafica de comparacion
methods = ["Por Defecto", "GridSearchCV", "Genetic Grid", "AG Directo (DEAP)"]
accuracies = [
    default_score * 100,
    grid_search.best_score_ * 100,
    genetic_grid.best_score_ * 100,
    best_ind.fitness.values * 100,
]

plt.figure(figsize=(9, 5))
bars = plt.bar(
    methods,
    accuracies,
    color=["#8c8c8c", "#1f77b4", "#ff7f0e", "#2ca02c"],
)
plt.ylim(50, 100)
plt.ylabel("Precision (%)")
plt.title("Comparativa de Ajuste de Hiperparametros (Capitulo 8)")

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 1,
        f"{yval:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
