The initial training set `/data/train.csv` contains 4194 rows (one row for each segment) and 1496 columns (features).
A genetic algorithm with CatboostRegressor was applied for fitness evaluation to implement a feature selection. 
Based on the GA's results, 15 features were selected and trained the model using CatboostRegressor with default parameters.

### Project structure

    .
    ├── ...
    ├── data                    
        ├── train.csv           # Original training set decomposed into feature set
        ├── test.csv            # Testing signal decomposed into feature set
        └── results.csv         # Modeling results prepared for submission
    │── notebooks
        └── earthquake.ipynb    # Misc
    ├── earthquake
        ├── config.py           # Configuration parameters    
        ├── ga.py               # GA for feature selection
        ├── generator.py        # Feature engineering
        ├── submission.py       # Make prediction and prepare file for submission
        └── utils.py            # Helpers
    └── ...

The train and test sets in `data` directory contain only 200 of 1496 features to reduce the file size.

### How to run

Step 1: Clone the repository and install requirements:

```
    $ git clone https://github.com/u-dhwani/Natural-Disaster-Prediction.git
    $ cd earthquake-prediction
    $ python -m pip install -r req.txt
```
    
Step 2: Launch `ga.py` script (starts the genetic algorithm implementing feature selection) from project's root directory:

```
    $ python earthquake/ga.py
```

### Feature engineering

The initial acoustic signal is decomposed into segments with 150000 rows per segment, 
This suggests that the training dataset has 4194 rows. Features are calculated as aggregations over segments.    

### Baseline model

Before starting with the feature selection, feature importance is calculated and the baseline model is trained on the 15 most important features.

The model is trained using CatboostRegressor with default parameters and the performance is evaluated with a stratified KFold (5 folds) cross-validation. 

CatboostRegressor (without any tuning) trained on 15 features having highest importance score demonstrates mean average error.

### Feature selection

To avoid a potential overfitting, a genetic algorithm is used for feature selection. In the genetic context, we suppose that the list of features (without duplicates) is the chromosome, whereas each gene represents one feature.
`n_features` is the input parameter controlling the amount of genes in the chromosome. 

The population with 50 chromosomes is generated, where each gene is generated as a random choice from initial list of features (1496 features).
To accelerate the performance, the feature set used in the baseline model is also added to the population.   

# setting individual creator
creator.create('FitnessMin', base.Fitness, weights=(-1,))
creator.create('Individual', Chromosome, fitness=creator.FitnessMin)

# register callbacks
toolbox = base.Toolbox()
toolbox.register(
    'individual', init_individual, creator.Individual,
    genes=genes, size=n_features)
toolbox.register(
    'population', tools.initRepeat, list, toolbox.individual)

# raise population
pop = toolbox.population(50)
```

Standard two-point crossover operator is used for crossing two chromosomes. 

```python
toolbox.register('mate', tools.cxTwoPoint)
```

To implement a mutation, first, a random amount of genes are generated (> 1), which needs to be mutated, and then
mutate these genes in order that the chromosome doesn't contain two equal genes. 

The mutation operator must return a tuple.

For fitness evaluation, lightened version of CatboostRegressor is used with decreased number of iterations and 
increased learning rate. The fitness evaluator must also return a tuple. 

The elitism operator is registered to select best individuals to the next generation. The amount of the best 
individuals is controlling by the parameter `mu` in the algorithm. To prevent populations with many
duplicate individuals, the standard `selBest` operator is overwritten.

Finally, everything is put together and launch `eaMuPlusLambda` evolutionary algorithm. 
Here `cxpb=0.2` is set, the probability that offspring is produced by the crossover, and `mutpb=0.8`, 
the probability that offspring is produced by mutation. Mutation probability is intentionally increased 
to prevent a high occurrence of identical chromosomes produced by the crossover.   

As a result, we get the list of 15 best features selected into the model.

### Training

The default CatboostRegressor is applied again to the found feature set and obtain mean average error.