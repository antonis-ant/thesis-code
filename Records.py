import pandas as pd


class Records:
    def __init__(self):
        # Names of the datasets' output column titles
        self.y_col_titles = ['woolfr', 'blood', 'feet', 'pelt', 'fullGI', 'mesent', 'epipl', 'liver',
                             'spleen', 'pluck', 'head', 'warmcarc', 'kidney', 'KKCF', 'tail',
                             'coldcarc', 'mw%', 'WtBefDIS', 'LEG', 'CHUMP', 'LOIN', 'BREAST',
                             'BESTEND', 'MIDNECK', 'SHOULDER', 'NECK']
        # Init column names list for records dataframe
        self.df_cols = self._build_cols(self.y_col_titles)
        # Init records dataframe
        self.model_records = pd.DataFrame(columns=self.df_cols)

        # Set directory in which to save results csv file
        self.records_dir = "records\\"

    def _build_cols(self, y_cols):
        cols = ['Regressor',
                'Data preprocessing',
                'Average R2 score',
                'Average MAE score',
                'Average RMSE score',
                'Average MAPE score',
                'Average R2 train score',
                'Average MAE train score',
                'Average RMSE train score',
                'Average MAPE train score']
        # Append scores for each individual output variable to columns list
        for col_title in y_cols:
            cols.append(col_title + ' r2 score')
            cols.append(col_title + ' mae score')
            cols.append(col_title + ' rmse score')
            cols.append(col_title + ' mape score')
        for col_title in self.y_col_titles:
            cols.append(col_title + ' r2 train score')
            cols.append(col_title + ' mae train score')
            cols.append(col_title + ' rmse train score')
            cols.append(col_title + ' mape train score')

        return cols

    def add_records(self, cv_results, data_prep='none'):
        # For each model ran:
        for key in cv_results:
            # Init dictionary to hold results
            data_dict = {'Regressor': key, 'Data preprocessing': data_prep}

            # 1. Calculate cv average of average scores for test set
            avg_scores = cv_results[key]['avg_scores']
            avg_scores_len = len(avg_scores)
            data_dict['Average R2 score'] = sum(item['score_r2'] for item in avg_scores) / avg_scores_len
            data_dict['Average MAE score'] = sum(item['score_mae'] for item in avg_scores) / avg_scores_len
            data_dict['Average RMSE score'] = sum(item['score_rmse'] for item in avg_scores) / avg_scores_len
            data_dict['Average MAPE score'] = sum(item['score_mape'] for item in avg_scores) / avg_scores_len

            # 2. Calculate cv average of average scores for train set
            avg_train_scores = cv_results[key]['avg_train_scores']
            data_dict['Average R2 train score'] = sum(item['score_r2'] for item in avg_train_scores) / avg_scores_len
            data_dict['Average MAE train score'] = sum(item['score_mae'] for item in avg_train_scores) / avg_scores_len
            data_dict['Average RMSE train score'] = sum(item['score_rmse'] for item in avg_train_scores) / avg_scores_len
            data_dict['Average MAPE train score'] = sum(item['score_mape'] for item in avg_train_scores) / avg_scores_len

            # 3. Calculate cv average of raw scores for test set
            raw_scores = cv_results[key]['raw_scores']
            raw_scores_len = len(raw_scores)
            cv_avg_raw_r2 = sum(item['score_r2'] for item in raw_scores) / raw_scores_len
            cv_avg_raw_mae = sum(item['score_mae'] for item in raw_scores) / raw_scores_len
            cv_avg_raw_rmse = sum(item['score_rmse'] for item in raw_scores) / raw_scores_len
            cv_avg_raw_mape = sum(item['score_mape'] for item in raw_scores) / raw_scores_len

            # 4. Calculate cv average of raw scores for train set
            raw_train_scores = cv_results[key]['raw_train_scores']
            cv_avg_raw_r2_train = sum(item['score_r2'] for item in raw_train_scores) / raw_scores_len
            cv_avg_raw_mae_train = sum(item['score_mae'] for item in raw_train_scores) / raw_scores_len
            cv_avg_raw_rmse_train = sum(item['score_rmse'] for item in raw_train_scores) / raw_scores_len
            cv_avg_raw_mape_train = sum(item['score_mape'] for item in raw_train_scores) / raw_scores_len

            # 5. Unfold individual average raw scores
            for i in range(len(self.y_col_titles)):
                # test scores
                data_dict[self.y_col_titles[i] + ' r2 score'] = cv_avg_raw_r2[i]
                data_dict[self.y_col_titles[i] + ' mae score'] = cv_avg_raw_mae[i]
                data_dict[self.y_col_titles[i] + ' rmse score'] = cv_avg_raw_rmse[i]
                data_dict[self.y_col_titles[i] + ' mape score'] = cv_avg_raw_mape[i]
                # train scores
                data_dict[self.y_col_titles[i] + ' r2 train score'] = cv_avg_raw_r2_train[i]
                data_dict[self.y_col_titles[i] + ' mae train score'] = cv_avg_raw_mae_train[i]
                data_dict[self.y_col_titles[i] + ' rmse train score'] = cv_avg_raw_rmse_train[i]
                data_dict[self.y_col_titles[i] + ' mape train score'] = cv_avg_raw_mape_train[i]

            # Append results to records dataframe for better presentation
            self.model_records = self.model_records.append(data_dict, ignore_index=True)

        # print(self.model_records)

    def get_last_rec(self):
        return self.model_records.tail(1)

    def get_records(self):
        return self.model_records

    def export_records_csv(self, filename):
        path = self.records_dir + filename
        self.model_records.to_csv(path, index=False)

# r = Records()
