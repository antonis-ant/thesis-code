import pandas as pd


class Records:
    def __init__(self):
        # Set record columns
        cols = ['Regressor',
                'Data_preprocessing',
                'Average R2 test score',
                'Average MAE test score',
                'Average RMSE test score',
                'Average MAPE test score',
                'Average R2 train score',
                'Average MAE train score',
                'Average RMSE train score',
                'Average MAPE train score',
                'Average test time',
                'Average train time']

        # Init records dataframe
        self.model_records = pd.DataFrame(columns=cols)

        # A subset of the record columns used for simpler presentation
        self.cols_compact = ['Regressor',
                             'Data_preprocessing',
                             'Average R2 test score',
                             'Average MAE test score',
                             'Average RMSE test score',
                             'Average MAPE test score',
                             'Average test time',
                             'Average train time']

        # Set directory in which to save results csv file
        self.save_dir = "records\\"

    def add_record(self, cv, cv_results, data_prep='none'):
        # Extract & prepare all useful data
        n_splits = cv.get_n_splits()
        # Get average test scores
        test_score_r2_avg = sum(cv_results['test_score_r2']) / n_splits
        test_score_mae_avg = sum(cv_results['test_score_mae']) / n_splits
        test_score_rmse_avg = sum(cv_results['test_score_rmse']) / n_splits
        test_score_mape_avg = sum(cv_results['test_score_mape']) / n_splits
        # Get average train scores
        train_score_r2_avg = sum(cv_results['train_score_r2']) / n_splits
        train_score_mae_avg = sum(cv_results['train_score_mae']) / n_splits
        train_score_rmse_avg = sum(cv_results['train_score_rmse']) / n_splits
        train_score_mape_avg = sum(cv_results['train_score_mape']) / n_splits
        # Get average fit & test times
        test_time_avg = sum(cv_results['score_time']) / n_splits
        train_time_avg = sum(cv_results['fit_time']) / n_splits

        # Organize data record in a dictionary
        data_dict = {
            'Regressor': cv_results['estimator'][0],
            'Data_preprocessing': data_prep,
            'Average R2 test score': test_score_r2_avg,
            'Average MAE test score': test_score_mae_avg,
            'Average RMSE test score': test_score_rmse_avg,
            'Average MAPE test score': test_score_mape_avg,
            'Average R2 train score': train_score_r2_avg,
            'Average MAE train score': train_score_mae_avg,
            'Average RMSE train score': train_score_rmse_avg,
            'Average MAPE train score': train_score_mape_avg,
            'Average test time': test_time_avg,
            'Average train time': train_time_avg
        }

        # Append model data record to dataframe
        self.model_records = self.model_records.append(data_dict, ignore_index=True)

        # Return current record
        # return self.model_records[self.cols_compact].tail(1)

    def get_last_rec(self, compact=True):
        if compact:
            return self.model_records[self.cols_compact].tail(1)
        return self.model_records.tail(1)

    def get_records(self, compact=True):
        if compact:
            return self.model_records[self.cols_compact]
        return self.model_records

    def export_records_csv(self, filename):
        path = self.save_dir + filename
        self.model_records.to_csv(path, index=False)

# r = Records()
