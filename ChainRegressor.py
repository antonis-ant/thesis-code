import pandas as pd


class ChainRegressor:
    """
    Implements a custom chain regressor that allows for different models to be used on each phase of the chain.
    """
    def __init__(self, chain_links):
        """
        The "chain links" must be provided on initialization and properly formatted as explained below in order for the
        regressor to work.

        @param chain_links: an array of tuples where each tuple contains the model and the corresponding dependent
         variables we want to predict with it.
         For example: chain_links = [(LinearRegression(), ['dependent_var1', 'dependent_var2',..]), ([..., ...]), ...]
        """
        self.chain_links = chain_links[:]
        self.original_cols = None

    def fit(self, X, y):
        """
        Fit the regression chain model on the dataset.

        @param X: independent variables
        @param y: dependent variables
        @return: the tuple containing the now trained models and the features they are trained for.
        """
        Xi = X.loc[:]
        # Grab the original columns to maintain the same order on the predicted variables
        self.original_cols = y.columns.tolist()
        for model, cols in self.chain_links:
            yi = y[cols]
            model.fit(Xi, yi)
            Xi = pd.concat([Xi, yi], axis=1)

        return self.chain_links

    def predict(self, X):
        """
        Make predictions with regressor chain model.

        @param X: independent variables
        @param y: dependent variables
        @return: an numpy array containing all predictions.
        """
        predictions = pd.DataFrame()
        Xi = X.loc[:]
        for model, cols in self.chain_links:
            # Get prediction of current model
            pred = model.predict(Xi)
            # Prepare predictions to use on next model
            yi = pd.DataFrame(pred, columns=cols)
            # Save current predictions to predictions dataframe
            predictions = pd.concat([predictions, yi], axis=1)
            # Append predictions to next model's features
            Xi = pd.concat([Xi, yi], axis=1)
        # Rearrange predictions dataframe to original column order
        predictions = predictions[self.original_cols]

        # Return predictions as numpy array for compatibility with scikit-learn functions
        return predictions.to_numpy()

