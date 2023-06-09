import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FN.utils.transform_dataset import transform_dataset,transform_dataset_credit,transform_dataset_census

def transform(dataset, df, valid):
    """
    given an original dataframe (COMPAS, Credit, or Census) function applies preprocessing used in the original paper.
    selects only numerical columns and returns them.
    """

    if valid:
        df = list(df.values())[0]

    if dataset == 'compas':
        df_binary, Y, S, Y_true = transform_dataset(df)
        columns_compas = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                          'days_b_screening_arrest', 'c_jail_time', 'date_diff_in_jail']
        df_cont = df_binary.iloc[:, 7:14]

        df_cont.columns = columns_compas

    elif dataset == 'credit':
        df_binary, Y, S, Y_true = transform_dataset_credit(df)
        # 5 because some columns had only 4 possible values (0,1/3,2/3,1) + nan as a 5th value.
        cont_columns = [i for i in range(len(df_binary.columns)) if len(df_binary.iloc[:, i].unique()) > 5]
        cont_column_names = [f'unknown_{i}' for i in range(len(cont_columns))]
        df_cont = df_binary.iloc[:, cont_columns]
        df_cont.columns = cont_column_names

    else:
        df_binary, Y, S, Y_true = transform_dataset_census(df)
        cont_columns = [i for i in range(len(df_binary.columns)) if len(df_binary.iloc[:, i].unique()) > 2]
        cont_column_names = [df_binary.columns[i] for i in cont_columns]
        cont_column_names[0] = 'age'
        df_cont = df_binary.iloc[:, cont_columns]

        df_cont.columns = cont_column_names

    return df_cont

def get_stats(dataset, df_cont, valid, part):
    """
    Given the dataframe returned by transform function. Function calculates mean and standard deviations of each column
    and returns this information in the form of dataframe.
    """

    df_fin = df_cont.mean().to_frame().reset_index()

    df_fin.columns = ['column', 'mean']

    df_fin['std'] = list(df_cont.std())

    df_fin['proper'] = valid

    df_fin['part'] = part

    df_fin['dataset'] = dataset

    print('Done')

    return df_fin

def preprocessing_function_original(df,dataset,stats):
    """
    Function is used to obtain statistics (mean and std) of datasets preprocessed in the original way.
    Original way in this case means that dataset was first preprocessed and only then split into train, test, and valid.
    """
    df_cont = transform(dataset, df, False)
    split = [int(0.7*len(df_cont)), int(0.8 * len(df_cont))]  # Train, validation, test
    train_dataset, val_dataset, test_dataset = np.split(df_cont.sample(frac=1, random_state=1), split)

    for df in [{"train":train_dataset}, {"val":val_dataset}, {"test":test_dataset}]:
        df_stats = get_stats(dataset, list(df.values())[0], False, list(df.keys())[0])
        stats = pd.concat([stats,df_stats])

    return stats

def preprocessing_function_correct(df,dataset, stats):
    """
    Function is used to obtain statistics (mean and std) of datasets preprocessed in the correct way.
    Correct way in this case means that dataset was split into train, test, and valid and then preprocessed.
    """

    split = [int(0.7*len(df)), int(0.8 * len(df))]  # Train, validation, test
    train_dataset, val_dataset, test_dataset = np.split(df.sample(frac=1, random_state=1), split)

    for df in [{"train":train_dataset}, {"val":val_dataset}, {"test":test_dataset}]:
        df_cont = transform(dataset, df, True)
        df_stats = get_stats(dataset, df_cont, True, list(df.keys())[0])
        stats = pd.concat([stats,df_stats])



    return stats

def plot(statistic, df):
    """
    After statistics for all the datasets are obtained, this function is used to plot the results.
    :param statistic: in this case it can be mean and std. If mean is passed standard deviation in means is calculated.
    If std is passed standard deviation in standard deviations of continuous columns is calculated.
    """

    plt.plot(df[df['proper'] == False]['column'], df[df['proper'] == False][statistic], label = 'Original Preprocessing')
    plt.plot(df[df['proper'] == True]['column'], df[df['proper'] == True][statistic], label = 'Right Preprocessing')

    if statistic == 'std':
        statistic = 'Standard Deviations'

    else:
        statistic = 'Means'

    plt.xticks(rotation=45)
    plt.title(f'Standard Deviation in {statistic} of Continuous Columns', size = 16, weight = 'bold')
    plt.ylabel('Standard Deviation', size = 12, weight = 'bold')
    plt.xlabel('Column Names', size = 12, weight = 'bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

def run(dataset,inputpath,stats):

    if dataset=='credit':
        df=pd.read_csv(inputpath,sep=' ')
    else:
        df=pd.read_csv(inputpath)

    stats_org = preprocessing_function_original(df,dataset, stats)
    stats = preprocessing_function_correct(df, dataset, stats_org)

    return stats


if __name__ == '__main__':

    df = pd.DataFrame()
    for i in [{'compas':'../data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv'},
     {'census':'../data/Census/adult'}, {'credit':'../data/Credit/german_credit'}]:
        stats = run(dataset=list(i.keys())[0],inputpath=list(i.values())[0], stats = df)
        df = pd.concat([df,stats]).drop_duplicates()

    df = df.groupby(by=['column', 'proper'])['mean', 'std'].std().reset_index()

    plot('mean', df)
    plot('std', df)