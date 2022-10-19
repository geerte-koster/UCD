"""
DESCRIPTION:
    Code to analyse the multiple choice and open answers to the mini-game evaluation questions. Two types of figures
    are made: the distributions of answers to closed questions and a countplot of categorised answers to the open questions.

AUTHOR:
    Geerte Koster
    email: geertekoster@hotmail.com
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def import_data(dataset, questions, ques_codes):
    """
    Function to import mini-game evaluation data and add information about the questions and answers options. Function takes the
    original data as input and returns the data with numerical values replaced by the categories that they represent.
    """
    data = pd.read_csv(dataset, sep=';', encoding='ANSI')

    data_q = data.loc[:, ques_codes]

    # create a dict that can replace the numerical values with categories in English
    repl_eng = {'s_1': {1: '<18', 2: '18-29', 3: '30-64', 4: '65+'},
                's_2': {1: 'Humanities', 2: 'Law', 3: 'Beta',
                        4: 'Medical', 5: 'Psychological', 6: 'Social Sciences',
                        7: 'Art, Music & Design', 8: 'Other'},
                's_3': {1: 'Yes', 2: 'No'},
                's_16': {1: 'too easy', 2: 'too difficult'},
                's_18': {1: 'too short', 2: 'too long'},
                's_6': {1: 'raw data', 2: 'analytical data', 3: 'functional data'}}

    # replace the numeric values with categories
    data_labs = data_q.replace(repl_eng)

    # open file containing the questions in English
    ques_eng = pd.read_csv(questions, sep=';', header=None)
    ques_eng["keyword"] = ["Age", "Faculty", "IT student", "Learning", "Clear goals", "Concentration", "Feedback",
                           "Knowledge increase",
                           "Apply knowledge", "Curious for more", "Control", "Skills", "Level mini-game", "Challenge",
                           "Time of mini-game",
                           "Most difficult data type", "Improvement"]

    return data_labs, ques_eng

def plot_figure(ques_codes, df, ques_eng, output, hue):
    """
    Function to plot the answers to the evaluation questions. Function takes the data set with categories as input and returns a figure
    with the distributions to the different answers.
    """
    # define which questions have likert scale
    likert = ["s_5", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17"]

    # dictionary with positions for subplots
    plot_num = {'s_1': 1, 's_2': (2, 3), 's_3': 4, 's_5': 5, 's_9': 6, 's_10': 7, 's_11': 9, 's_12': 10, 's_13': 11,
                's_14': 8, 's_15': 13, 's_16': 14, 's_17': 15, 's_18': 16, 's_6': 12}

    # create figure with space for 4x4 subplots and small font sizes + making pretty
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 8}

    # set formatting and layout settings
    matplotlib.rc('font', **font)
    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches(8.27, 11.69)
    loc = "left"
    size = 10
    title_y = 1.07

    # create subplots based on the type of questions and number of answering options
    for ques in ques_codes:
        title = ques_eng.loc[(ques_eng[0] == ques), "keyword"].item()
        ax = fig.add_subplot(4, 4, plot_num[ques])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        if ques in ["s_2", "s_6"]:
            c = sns.countplot(data=df, x=ques, ax=ax, hue=hue).set(xlabel=None)
            plt.legend([], [], frameon=False)
            ax.set_title(title, loc=loc, size=size, y=title_y)
            ax.tick_params(labelsize=5, labelrotation=20)
            # add numbers on top of bars in barplot
            for i in ax.containers:
                ax.bar_label(i, )

        elif ques in likert:
            c = sns.histplot(data=df, x=ques, discrete=True, ax=ax, hue=hue).set(
                xlabel=None)  # , hue="s_3", stat="probability"
            plt.legend([], [], frameon=False)
            plt.xticks([1, 2, 3, 4, 5])  # Set label locations.
            ax.set_title(title, loc=loc, size=size, y=title_y)

            for i in ax.containers:
                ax.bar_label(i, )

            if hue == None:
                q_mean = df[ques].mean()
                plt.axvline(x=q_mean, color='black', ls='--')
                plt.text(x=q_mean, y=ax.get_ylim()[1] + 2, s=round(q_mean, 2), color='red', ha="center")

            else:
                means = df.groupby(hue).mean()

                for mean in means[ques]:
                    plt.axvline(x=mean, color='black', ls='--')
                    plt.text(x=mean, y=ax.get_ylim()[1] + 0.7, s=round(mean, 2), color='red', ha="center")

        else:
            c = sns.countplot(data=df, x=ques, ax=ax,  hue=hue).set(xlabel=None)
            plt.legend([], [], frameon=False)
            ax.set_title(title, loc=loc, size=size, y=title_y)

            for i in ax.containers:
                ax.bar_label(i, )

    # adjust spacing
    wspace = 0.4  # the amount of width reserved for blank space between subplots
    hspace = 0.4  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.savefig(f'{output}.eps', format='eps')

def analyze_open_questions(coded_data):
    """
    Function to create bar plots from the categorised data made from the open answers of the mini-game evaluation.
    """
    coded_answers = pd.read_excel(coded_data)
    coded_answers = coded_answers.drop(['Unnamed: 6', 'Unnamed: 7'], axis=1)

    # dict containing everything for Q4 and Q15
    ans_dict = {"Q4":{"title":"Learning"}, "Q15":{"title":"Improvements"}}

    # save all coded answers
    ans_dict["Q4"]["coded_answers"] = coded_answers.iloc[:,1:3]
    ans_dict["Q15"]["coded_answers"] = coded_answers.iloc[:, 4:6]

    # save the used codes and descriptions
    ans_dict["Q4"]["codes"] = pd.read_excel(coded_data, sheet_name="CODES", usecols=[0,1])
    ans_dict["Q15"]["codes"] = pd.read_excel(coded_data, sheet_name="CODES", usecols=[2, 3])
    # drop rows with NaN for Q4
    ans_dict["Q4"]["codes"] = ans_dict["Q4"]["codes"].dropna()

    # plot the number of times a certain code / answer appears
    # set font style / size
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)

    fig = plt.figure()
    fig.set_size_inches(8.27, 4)

    for i, ques in enumerate(["Q4", "Q15"]):
        # split columns with multiple codes
        answers_split = ans_dict[ques]["coded_answers"][f'{ques}_C'].str.split('; ')
        answers_expanded = answers_split.explode(f"{ques}_C")

        ax = fig.add_subplot(1, 2, i+1)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        sns.countplot(y=answers_expanded, order=ans_dict[ques]["codes"][f"{ques}_CODES"], ax=ax, orient="h").set(ylabel=None, title=ans_dict[ques]["title"])

        # add number on top of bar
        for x in ax.containers:
            ax.bar_label(x)

        # replace x ticks by descriptions
        ax.set_yticks(range(len(ans_dict[ques]["codes"][f"{ques}_DESCRIPTION"])), ans_dict[ques]["codes"][f"{ques}_DESCRIPTION"])

    # adjust spacing
    wspace = 0.9  # the amount of width reserved for blank space between subplots
    left = 0.175  # the left side of the subplots of the figure
    fig.subplots_adjust(left=left, wspace=wspace)
    fig.savefig('open_questions.eps', format='eps')
    fig.savefig('open_questions.png', format='png')

def permutation_test(stat_df, prob_df):
    """
    Function to do 1000 permutation test to check the differences in mean scores for the different age groups and the IT degree groups
    (yes or no). Function returns distribution of differences in mean over 1000 permutation tests and corresponding p-values.
    """

    np.seed = 11
    ## prepare data with 1-5 scale answers
    # check the effect of shuffling the age labels
    age_df = stat_df.drop("IT student", axis=1)
    age_prob = prob_df.drop("IT student", axis=1)
    # check the effect of shuffling the IT student labels
    IT_df = stat_df.drop("Age", axis=1)
    IT_prob = prob_df.drop("Age", axis=1)

    stat_dict = {"Age":[age_df, age_prob], "IT student":[IT_df, IT_prob]}
    df_stats = pd.DataFrame(index=["Age", "IT student"], columns=stat_df.drop(["IT student", "Age"], axis=1).columns)
    prob_stats = pd.DataFrame(index=["Age", "IT student"], columns=prob_df.drop(["IT student", "Age"], axis=1).columns)

    for cat in ["Age", "IT student"]:
        fig, axs = plt.subplots(3,3, sharex=True)
        fig.supxlabel('difference in mean of two groups')
        fig.supylabel('counts')
        fig.suptitle(f"Permutation test for {cat} groups")

        df = stat_dict[cat][0]
        mean_df = df.groupby(cat).mean()

        for num, question in enumerate(df.drop(cat, axis=1).columns):
            # for each question, check difference in mean between groups before and after permuation
            sample_stat = mean_df.loc[df[cat].unique()[0],question] - mean_df.loc[df[cat].unique()[1],question]

            stats = np.zeros(1000)
            for k in range(1000):
                # shuffle the labels for the groups
                labels = np.random.permutation((df[cat] == df[cat].unique()[0]).values)
                # calculate difference in means of groups as defined by new labels
                stats[k] = np.mean(df.loc[labels, question]) - np.mean(df.loc[labels == False, question])

            # determine p-value by checking how many times the permutated difference in mean is bigger than the real diff in mean
            p_value = np.mean(abs(stats) > abs(sample_stat))
            df_stats.loc[cat,question] = p_value

            # plot the permutation test
            ax = axs.ravel()[num]
            ax.hist(stats, label='Permutation Statistics', bins=30)
            ax.axvline(x=sample_stat, c='r', ls='--', label='Sample Statistic')
            ax.set_title(question)
            wspace = 0.4  # the amount of width reserved for blank space between subplots
            hspace = 0.4  # the amount of height reserved for white space between subplots
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        fig.savefig(f'{cat}_perm.eps', format='eps')

        ## same procedure but now with categorical questions 2 categories
        fig, axs = plt.subplots(2,1, sharex=True)
        fig.supxlabel('difference in proportions of two groups')
        fig.supylabel('counts')
        fig.suptitle(f"Permutation test for {cat} groups")

        df = stat_dict[cat][1]

        for num, question in enumerate(df.drop(cat, axis=1).columns):
            df = stat_dict[cat][1]
            # drop NA columns
            df = df.loc[:, [cat, question]].dropna()
            # convert categories to dummy 0 and 1
            replace_dict = {df[question].unique()[0]:0, df[question].unique()[1]:1}
            df = df.replace(replace_dict)

            mean_df = df.groupby(cat).mean()

            # for each question, check difference in mean between groups before and after permuation
            sample_stat = mean_df.loc[df[cat].unique()[0],question] - mean_df.loc[df[cat].unique()[1],question]

            stats = np.zeros(1000)
            for k in range(1000):
                # shuffle the labels for the groups
                labels = np.random.permutation((df[cat] == df[cat].unique()[0]).values)
                # calculate difference in means of groups as defined by new labels
                stats[k] = np.mean(df.loc[labels, question]) - np.mean(df.loc[labels == False, question])

            # determine p-value by checking how many times the permutated difference in mean is bigger than the real diff in mean
            p_value = np.mean(abs(stats) > abs(sample_stat))
            prob_stats.loc[cat,question] = p_value

            # plot the permutation test
            ax = axs.ravel()[num]
            ax.hist(stats, label='Permutation Statistics', bins=30)
            ax.axvline(x=sample_stat, c='r', ls='--', label='Sample Statistic')
            ax.set_title(question)
            wspace = 0.4  # the amount of width reserved for blank space between subplots
            hspace = 0.4  # the amount of height reserved for white space between subplots
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        fig.savefig(f'{cat}_perm_props.eps', format='eps')

    return df_stats, prob_stats

def main():

    ## analyse the quantative data
    dataset = "data/dataset.csv"
    questions = "data/questions_eng.csv"

    # define the columns that contain answers to MPC questions
      # s_4 and s_7 are open questions
    ques_codes = ['s_1', 's_2', 's_3', 's_5', 's_9', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17',
                  's_18', 's_6']

    df, ques_eng =  import_data(dataset, questions, ques_codes)

    # create figure for each category
    plot_figure(ques_codes, df, ques_eng, output="all_groups", hue=None)

    # (optional figures)
    # # stratify by age
    ##plot_figure(ques_codes, df, ques_eng, output="age", hue="s_1")
    #
    # # stratify by IT
    # plot_figure(ques_codes, df, ques_eng, output="IT", hue="s_3")
    #
    # #stratify by faculty
    # plot_figure(ques_codes, df, ques_eng, output="faculty", hue="s_2")

    # do permutation testing of mean values

    # create pandas df for statistical testing + set keywords as column titels
    # create rename dictionary to set keywords as titels
    name_dict = dict(zip(ques_eng[0], ques_eng["keyword"]))

    stat_questions = ["s_1", "s_3", "s_5", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17"]
    stat_df = df.loc[:,stat_questions]
    stat_df = stat_df.rename(columns=name_dict)

    prop_questions = ["s_1", "s_3", "s_16", "s_18"]
    prop_df = df.loc[:,prop_questions]

    prop_df = prop_df.rename(columns=name_dict)

    # calculate means for different groups (optional save to excel)
    mean_age = stat_df.groupby("Age").mean()
    mean_IT = stat_df.groupby("IT student").mean()
    #mean_age.to_excel("mean_age.xlsx", float_format="%.3f")
    #mean_IT.to_excel("mean_IT.xlsx", float_format="%.3f")

    # do the permutation testing (optional save output to excel)
    df_statistics, prob_stats = permutation_test(stat_df, prop_df)
    print(prob_stats)
    #df_statistics.to_excel("df_statistics.xlsx")

    ## analyse qualitative data
    coded_data = "data/Analysis_open_questions.xlsx"
    analyze_open_questions(coded_data)

if __name__ == '__main__':
    main()