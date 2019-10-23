"""
This module contains the core functions used for data visualization and
summaries.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import missingno as msno
import scripts.heatmap as heatmap
sns.set

def plot_scatter(data=None, x_name="", y_name="", x=None, y=None, title="Scatter Plot",
                    definitions=None, hue=None):
    """
    Plots two flat arrays as a scatter plot given a dataframe. If a hue is provided, the plot
    will be color-coded based on the categorical data given by hue. If definitions are provided,
    the color coded legend will contain string names instead of numbers. The user can provide
    either two array like structures to plot against each other, or a dataframe with
    the two desired column names to plot against each other. Both of these instances
    cannot happen at the same time.

    Paramaters
        data: pandas DataFrame, The dataframe containing the data.
        x_name: string, The name of the first column to plot on the x-axis
        y_name: string, The name of the second column to plot on the y-axis
        x: array-like, An array of data that will be on the x-axis
        y: array-like, An array of data that will be on the y-axis
        title: string, The name of the scatter plot. Defaults to 'Scatter Plot'.
        definitions: dictionary, A dictionary that maps integer categories to meaningful string names.
        For example,
            sample_dict = {
                0: 'coal',
                1: 'gas',
                3: 'oil'
            }
        hue: string, The column name of the categorical data to be used as a color code.

    Return
        Graphical object

    Usage
        from mlvizer import vizer as vz
        data = load_data() # your data
        fuel_definitions = {
            0: 'coal'
            1: 'gas'
            2: 'oil'
        }

        vz.plot_scatter(data, 'generation', 'co2_emissions',
                        title='Generation vs CO2 Emissions', definitions=fuel_definitions,
                        hue='fuel_type')

    Notes
        If the user wants a color coded plot based on the hue, and the data contained in hue
        are already string names, it is not necessary to provide definitions. If the user provides
        definitions but no hue, nothing happens and the program will continue like normal.
    """
    title_offset = 18 # arbitrary, seems to be the best fit

    # makes sure the user only chooses one of the methods to plot
    try:
        if not (data is None):
            assert x is None and y is None
            if definitions:
                if not hue:
                    raise("Error: hue is missing.")
                # if definitions are provided and hue is already in correct format, continue
                if hue:
                    hue = data[hue].apply(lambda x: definitions[x])
                    plt.legend(loc='best')
            x_col = data[x_name]
            y_col = data[y_name]
            plt.scatter(x_col, y_col, c=hue)


        elif x and y:
            assert data is None
            plt.scatter(x, y)

    except:
        print("Argument Error: Make sure you only provide data, or only x and y.")

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if x_name and y_name:
        plt.title(x_name + " vs " + y_name, pad=title_offset)
    else:
        plt.title(title, pad=title_offset)
    plot = plt.show()
    print('\n')

    return plot

def get_num_missing(data):
    """
    Counts the number of missing data in the dataset with respect to each variable. Missing is
    defined to be NaN or some related None value.

    Paramaters
        data: pandas DataFrame

    Return
        List of counts
    """

    max_examples = len(data.index)
    counts_present = data.count()
    return [max_examples - present for present in counts_present]

def print_stats(data):
    """
    Prints the number of missing values, the number of examples, number of features, mean, mode
    min, and max.

    Parameters
        data: pandas DataFrame
    """
    # set_trace()

    stats = {
        'num_present': data.count(),
        # 'num_missing': get_num_missing(data),
        'max': data.max(),
        'min': data.min(),
        'mean': data.mean(),
        'median': data.median(),
    }

    print(pd.DataFrame(data=stats), '\n')

def summary(data, target_name, definitions=None, hue=None):
    """
    Prints a full set of statistics and visuals for a data set, including missing values.
    The target_name specifies the column name of the feature considered to be the dependent variable.
    This will be used to compare against each of the other features.

    Parameters
        data: pandas DataFrame
        target_name: string, The column name of the target variable.
    """
    # print number of examples and features
    print("Number of data points: ", len(data.index))
    print("Number of features: ", len(data.columns))
    print('\n')

    print_stats(data)
    print('\n')

    # print('Missing values')
    # msno.matrix(data)
    # print('\n')

    # plot scatter for each of the features against target_name
    for col_name in data.columns:
        if col_name == target_name:
            continue
        plot_scatter(data=data, x_name=col_name, y_name=target_name,
                    definitions=definitions, hue=hue)

    # plot informational correlation plot
    heatmap.corrplot(data)

""" Machine Learning Visualization Tools"""

def plot_prediction_scatter(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()

# buggy
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true))) * 100
    return mape

def print_error_stats(y_true, y_pred):
    """
    Prints error stats given the predicted values vs the actual values.
    """
    print("Coefficient of Determination:", metrics.r2_score(y_true, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_true, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_true, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    # MAPE is buggy, so exlcuding this from metrics for now
#     print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_true, y_pred))
    print('\n')

def plot_train_test_errors(y_train, y_pred_train, y_test, y_pred_test):
    """
    Plots the training and test errors for a regression model as bar graphs. This will
    allow the user to see any over-fitting, variance, or bias the model may have.

    Parameters
    y_train: pd.Series, the true train values
    y_pred_train: pd.Series, the predicted values from x_train by model fitted onto the training set
    y_test: pd.Series, the true test values
    y_pred_test: pd.Series, the predicted test values from x_test by the model fitted onto the training set

    References
    https://pythonspot.com/matplotlib-bar-chart/
    """

    # train stats
    train_cod = metrics.r2_score(y_train, y_pred_train)
    train_mae = metrics.mean_absolute_error(y_train, y_pred_train)
    train_mse = metrics.mean_squared_error(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)

    # test stats
    test_cod = metrics.r2_score(y_test, y_pred_test)
    test_mae = metrics.mean_absolute_error(y_test, y_pred_test)
    test_mse = metrics.mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)

    # set up label variables
    objects = ('Train', 'Test')
    y_pos = [0,0.3] # arbitrary selection

    # plot cod comparison
    cod_performance = [train_cod, test_cod]
    plt.bar(y_pos, cod_performance, align='center', alpha=0.5, width=0.2)
    plt.xticks(y_pos, objects)
    plt.ylabel('COD')
    plt.title('Coefficient of Determination')
    plt.show()

    # plot mse comparison
    mse_performance = [train_mse, test_mse]
    plt.bar(y_pos, mse_performance, align='center', alpha=0.5, width=0.2)
    plt.xticks(y_pos, objects)
    plt.ylabel('Error')
    plt.title('Mean Squared Error')
    plt.show()

    # plot mae comparison
    mae_performance = [train_mae, test_mae]
    plt.bar(y_pos, mae_performance, align='center', alpha=0.5, width=0.2)
    plt.xticks(y_pos, objects)
    plt.ylabel('Error')
    plt.title('Mean Absolute Error')
    plt.show()

    # plot rmse comparison
    rmse_performance = [train_rmse, test_rmse]
    plt.bar(y_pos, rmse_performance, align='center', alpha=0.5, width=0.2)
    plt.xticks(y_pos, objects)
    plt.ylabel('RMSE')
    plt.title("Root Mean Square Error")
    plt.show()

def plot_prediction_bar_graph(y_true, y_pred, num_samples=40, title=""):
#     set_trace()

    compare_df_train = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred.flatten()})
    small_compare_train = compare_df_train.head(num_samples)
    small_compare_train.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title(title)
    plt.show()

# Terry: added a scaler parameter for inverse transformation, but need to make sure pass in the original y_train and y_test (not scaled) for consistency
def model_summary(model, x_train, x_test, y_train, y_test, num_samples=40, scaler = None):
    """
    Given a model, summarizes performance by comparing true target values
    vs predicted target values

    Parameters
    model: model-like, can be a keras or scikit linear regression model object, must be already trained
    x: array-like, the training data
    y_true: array-like, the true target
    num_samples: int, subset of data points to plot to get a grasp of model performance
    """
    # From Terry
    if scaler != None:
        print("You are passing in a scaler for inverse transform. Make sure the y_trian and y_test are original values and not scaled")

    # predict target values
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    if scaler != None:
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)

    # plot and print training info
    print("Training Set Stats")
    print_error_stats(y_train, y_train_pred)
    plot_prediction_bar(y_train, y_train_pred, num_samples=num_samples, title="Training Set Predictions")
    plot_prediction_scatter(y_train, y_train_pred)
    print("\n")

    # plot and print test info
    print("Test Set Stats")
    print_error_stats(y_test, y_test_pred)
    plot_prediction_bar(y_test, y_test_pred, num_samples=num_samples, title="Test Set Predictions")
    plot_prediction_scatter(y_test, y_test_pred)
    print("\n")

    # compare Training and Test Errors
    plot_train_test_errors(y_train, y_train_pred, y_test, y_test_pred)

# def compare_models(x_test, y_test, **kwargs):

# #     predictions = {model_name:predictions for}
#     for model_name, model in kwargs.items():

#         # plot COD comparison

def plot_loss_error(history, loss):
    """
    Given a history object, plots the loss vs num_epochs.

    Parameters
    history: history object, a history object produced by fitting the model
    loss: string, The loss function as defined in keras
        Example: 'mean_squared_error'
    """

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = plt.figure()
    fig.title('Model Performance')
    fig.xlabel('Epoch')
    fig.ylabel(loss)
    fig.plot(hist['epoch'], hist[loss],
           label='Train Error')
    fig.plot(hist['epoch'], hist['val_' + loss],
           label = 'Val Error')
    fig.legend()
    fig.show()

    return fig # is this what I change?

def save_graphic(graphic, location):
    """
    Given a graphical object, saves the object to a specified location as a png

    Parameters
        graphic: graphic-like, A plot or a graph
        location: string, The file path to the directory where the image will be saved.
    """
    # TODO: implement this function
    pass

def plot_distribution(series):
    """ Given some array-like data, plots the distribution

    Parameters
        series: array-like, An array containing data.
    """
    # TODO
    pass
