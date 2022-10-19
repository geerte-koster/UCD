"""
DESCRIPTION:
    Code to create course and semester based visualisations of curriculum mappings as used in first mockup.
    These figures show the overlap of course and program learning outcomes (LOs).

AUTHOR:
    Geerte Koster
    email: geertekoster@hotmail.com
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import dataframe_image as dfi
from operator import itemgetter

def import_data(path, semesters, random_data):
    """Function that loads mapping files and returns one dataframe containing each course, semester, and a dictionary
    containing the mapping"""

    if random_data == "yes":
        # load the mapping of course PRPSYK100 and create randomized versions of it for the other courses

        # import data file for mapped course PRPSYK100
        P100_levels = pd.read_excel(path, sheet_name=None)
        P100_levels["department"] = P100_levels["department"].columns[0]

        # import file with course names and semesters
        course_df = pd.read_excel(semesters, dtype={"course_code":str, "semester":int})
        course_df = course_df.set_index("course_code")
        course_df["mapping"] = "X"

        # for now, set the mapping of every course to randomizing the columns of PRPSYK100
        for course in course_df.index:
            if course == "PRPSYK100":
                course_df.at[course, "mapping"] = P100_levels
            else:
                P_random = P100_levels.copy()
                K = ["K1", "K2", "K3", "K4", "K5", "K6"]
                random.shuffle(K)

                F = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
                random.shuffle(F)

                GK = ["GK1", "GK2", "GK3", "GK4", "GK5", "GK6", "GK7", "GK8", "GK9"]
                random.shuffle(GK)

                P_random["Kunnskaper"] = P_random["Kunnskaper"].drop("Læringsutbyte", axis=1)
                P_random["Kunnskaper"].columns = K

                P_random["Ferdigheter"] = P_random["Ferdigheter"].drop("Læringsutbyte", axis=1)
                P_random["Ferdigheter"].columns = F

                P_random["Kompetanse"] = P_random["Kompetanse"].drop("Læringsutbyte", axis=1)
                P_random["Kompetanse"].columns = GK

                course_df.at[course,"mapping"] = P_random
    else:
        # load the excel files of all courses
        # import file with course names and semesters
        course_df = pd.read_excel(semesters, dtype={"course_code":str, "semester":int})
        course_df = course_df.set_index("course_code")
        course_df["mapping"] = "X"

        for course in course_df.index:
            mapping = pd.read_excel(f"{path}{course}.xlsx", sheet_name=None)
            mapping["department"] = mapping["department"].columns[0]
            course_df.at[course, "mapping"] = mapping

    return course_df

def PLO_coverage(course_df, levels, view):
    # count for each course how often a PLO is covered
    # create an overview of coverage of PLOs for each level separately and store in dictionary
    level_dfs = {}
    rel_dfs = {}

    for level in levels:

        if view == "semester":
            # create a dictionary for each semester and learning outcome category containing zeros, here the coverage can be stored
            long_coverage = {"Kunnskaper": pd.DataFrame(0, index=range(1, 13), columns=["K1", "K2", "K3", "K4", "K5", "K6"]),
                             "Ferdigheter": pd.DataFrame(0, index=range(1, 13),
                                                         columns=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]),
                             "Kompetanse": pd.DataFrame(0, index=range(1, 13),
                                                        columns=["GK1", "GK2", "GK3", "GK4", "GK5", "GK6", "GK7", "GK8",
                                                                 "GK9"])}
            for PLO in long_coverage.keys(): # PLO is the LO category, so Kunnskaper etc.

                for course in course_df.index: #["PRPSYK100"]:

                    for LO in long_coverage[PLO].columns.values.tolist():

                        mapping = course_df.loc[course,"mapping"][PLO] # dataframe containing the mapping
                        if level in mapping[LO].unique():
                            # check if one of the levels reached in the mapping is the same as the current level being mapped
                            semester = course_df.loc[course, "semester"]

                            # if a match with the specified level is found, the coverage is increased with one
                            long_coverage[PLO].at[semester, LO] += 1

            level_dfs[level]= long_coverage
            # for the semester view, we don't need the relative and maximal coverage dfs, so they are assigned to "empty"
            rel_dfs = "empty"
            max_coverage = "empty"

        if view == "course":
            # create a dictionary for each course and learning outcome category containing zeros, here the coverage can be stored
            long_coverage = {
                "Kunnskaper": pd.DataFrame(0, index=course_df.index, columns=["K1", "K2", "K3", "K4", "K5", "K6"]),
                "Ferdigheter": pd.DataFrame(0, index=course_df.index,
                                            columns=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]),
                "Kompetanse": pd.DataFrame(0, index=course_df.index,
                                           columns=["GK1", "GK2", "GK3", "GK4", "GK5", "GK6", "GK7", "GK8",
                                                    "GK9"])}
            relative_coverage = {
                "Kunnskaper": pd.DataFrame(0, index=course_df.index, columns=["K1", "K2", "K3", "K4", "K5", "K6"]),
                "Ferdigheter": pd.DataFrame(0, index=course_df.index,
                                            columns=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]),
                "Kompetanse": pd.DataFrame(0, index=course_df.index,
                                           columns=["GK1", "GK2", "GK3", "GK4", "GK5", "GK6", "GK7", "GK8",
                                                    "GK9"])}

            for PLO in long_coverage.keys():  # PLO is the LO category, so Kunnskaper etc.

                for course in course_df.index:  # ["PRPSYK100"]:

                    for LO in long_coverage[PLO].columns.values.tolist():
                        mapping = course_df.loc[course, "mapping"][PLO]  # dataframe containing the mapping

                        if level in mapping[LO].unique(): # check if one of the levels reached in the mapping is the same as the current level being mapped
                            # check how many course LOs map to the specific level of the PLO
                            coverage = sum(mapping[LO] == level)
                            # if a match with the specified level is found, the coverage is increased with
                            long_coverage[PLO].at[course, LO] += coverage

                    ## calculate the relative coverage by dividing the mapping coverage by the number of CLOs for a certain PLO category
                    # calculate the number of CLO for a course per LO category
                    CLO_total = course_df.loc[course, "mapping"][PLO].shape[0]
                    # divide the coverage by the number of CLOs in a specific category
                    relative_coverage[PLO].loc[course, :] /= CLO_total

            rel_dfs[level] = relative_coverage
            level_dfs[level] = long_coverage

    # for the course view, we also want to check the highest level mapping
    if view == "course":
        max_coverage = {
            "Kunnskaper": pd.DataFrame(0, index=course_df.index, columns=["K1", "K2", "K3", "K4", "K5", "K6"]),
            "Ferdigheter": pd.DataFrame(0, index=course_df.index,
                                        columns=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]),
            "Kompetanse": pd.DataFrame(0, index=course_df.index,
                                       columns=["GK1", "GK2", "GK3", "GK4", "GK5", "GK6", "GK7", "GK8",
                                                "GK9"])}
        for PLO in max_coverage.keys():  # PLO is the LO category, so Kunnskaper etc.
            for course in course_df.index:  # loop over each course
                for LO in max_coverage[PLO].columns.values.tolist():
                    mapping = course_df.loc[course, "mapping"][PLO]  # dataframe containing the mapping
                    # find the maximal level reached for each PLO

                    max_level = np.max(mapping[LO])

                    # if max level is not Nan, it is added to the max coverage df
                    if np.isnan(max_level) == False:
                        max_coverage[PLO].at[course, LO] = max_level

    return level_dfs, rel_dfs, max_coverage

def transform_dfs(coverage, levels, view):
    # function to transform the coverage dfs to be able to plot the data
    plot_dfs = {}

    # create a df for each level
    for level in levels:
        plot_dfs[level] = {}
        # get the coverage df for the specific level
        coverage_level = coverage[level]

        # loop over the different PLO categories dfs
        for PLO in coverage_level.keys():
            PLO_df = coverage_level[PLO].copy()

            if view == "semester":
                PLO_df.index = PLO_df.index.rename("semester")
                PLO_df.reset_index(inplace=True)
                # melt the df to be able to create plots
                plot_df = PLO_df.melt(id_vars=["semester"], value_vars=PLO_df.columns.difference(["semester"]),
                                      var_name="PLO", value_name="coverage")


            elif view == "course":
                PLO_df.index = PLO_df.index.rename("course")
                PLO_df.reset_index(inplace=True)

                plot_df = PLO_df.melt(id_vars=["course"], value_vars=PLO_df.columns.difference(["course"]),
                                      var_name="PLO", value_name="coverage")

            # add the specific level to the melted df
            plot_df["level"] = level
            # save the df in the plot_dfs dictionary
            plot_dfs[level][PLO] = plot_df

    # create 3 dataframes for each PLO category by concatenating the dfs containing the different levels together
    final_level_dfs = {}
    level_df = plot_dfs[levels[0]]

    for i in range(len(levels)-1):
        for PLO in coverage_level.keys():
            final_level_dfs[PLO] = pd.concat([level_df[PLO], plot_dfs[levels[i+1]][PLO]], ignore_index=True)
            level_df[PLO] = final_level_dfs[PLO]

    return final_level_dfs

def bubble_plot(plot_dfs, view, level):
    if level == "all":
        for PLO in plot_dfs.keys():
            df = plot_dfs[PLO]
            fig = plt.figure()
            if view == "semester":
                sns.scatterplot(data=df, x="semester", y="PLO", size="coverage", sizes=(0, 500), hue="level", alpha=0.7,
                                palette=list(itemgetter(0,3,5)(sns.color_palette("viridis_r")))).set(title=PLO)
                plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
                plt.xlabel("Semester")
                plt.ylabel("Program learning outcomes")
                plt.xticks(range(1,13))

            elif view == "course":
                sns.scatterplot(data=df, x="course", y="PLO", size="coverage", sizes=(0, 500), hue="level", alpha=0.7,
                                palette=list(itemgetter(0,3,5)(sns.color_palette("viridis_r")))).set(title=PLO)
                plt.xticks(rotation=90)
                plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
                plt.xlabel("")
                plt.ylabel("Program learning outcomes")
    else:
        for PLO in plot_dfs.keys():
            df = plot_dfs[PLO][plot_dfs[PLO]["level"] == level]
            fig = plt.figure()
            if view == "semester":
                sns.scatterplot(data=df, x="semester", y="PLO", size="coverage", sizes=(0, 500), alpha=0.7,
                                color=itemgetter(0,3,5)(sns.color_palette("viridis_r"))[level-1]).set(title=PLO)
                plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
                plt.xlabel("Semester")
                plt.ylabel("Program learning outcomes")
                plt.xticks(range(1, 13))

            elif view == "course":
                sns.scatterplot(data=df, x="course", y="PLO", size="coverage", sizes=(0, 500), alpha=0.7,
                                color=itemgetter(0,3,5)(sns.color_palette("viridis_r"))[level-1]).set(title=PLO)
                plt.xticks(rotation=90)
                plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
                plt.xlabel("")
                plt.ylabel("Program learning outcomes")

def bar_plot(plot_dfs, view, selection, course_df):
    if view == "semester":
        fig, axes = plt.subplots(len(selection[view]), 3, sharey=True)
        fig.tight_layout()
    if view == "course":
        fig, axes = plt.subplots(len(selection[view]), 3)
        fig.tight_layout()

    for PLO_num, PLO in enumerate(plot_dfs.keys()):
        df = plot_dfs[PLO]

        for ax_num, i in enumerate(selection[view]):
            df_selection = df.loc[df[view] == i]

            sns.barplot(data=df_selection, x="PLO", y="coverage", hue="level", palette=itemgetter(0,3,5)(sns.color_palette("viridis_r")),
                        ax=axes[ax_num, PLO_num])
            axes[ax_num, PLO_num].set_title(f"{view} {i}")
            axes[ax_num, PLO_num].set_xlabel("")
            axes[ax_num, PLO_num].set_ylabel("number of CLOs that map to PLO")
            axes[ax_num, PLO_num].legend([], [], frameon=False)
            if view == "course":
                # calculate the number of CLO for a course per LO category
                CLO_total = course_df.loc[i, "mapping"][PLO].shape[0]
                axes[ax_num, PLO_num].set(ylim=(0,CLO_total))
                axes[ax_num, PLO_num].set_yticks(range(0, CLO_total+1))

            sns.despine(fig=fig, ax=axes[ax_num, PLO_num], top=True, right=True, left=False, bottom=False, offset=None,
                        trim=False)

    handles, labels = axes[ax_num, PLO_num].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

def table_view(max_coverage, levels, selection, output_folder):
    for PLO in max_coverage.keys():
        if selection == "all":
            df = max_coverage[PLO]

        else:
            df = max_coverage[PLO].loc[selection,:]

        courses = df.index
        ## also make tables with the row - columns reversed
        df_t = df.transpose()

        # ad a total column that indicates how many courses covered the LO in a certain level
        def total_coverage(row, level):
            level_sum = sum(row == level)
            return level_sum

        for level in levels:
            #df_t[f"Total level {level}"] = df_t.apply(total_coverage, args=[level], axis=1)
            df.loc[f"Total level {level}"] = df.loc[courses].apply(total_coverage, args=[level], axis=0)

        ## set the settings for a heatmap table visualisation
        df_style = df.style.background_gradient(subset=(df.index.isin(courses), df.columns), vmin=0, vmax=3)
        #df_t_style = df_t.style.background_gradient(subset=(df.index.isin(courses)), cmap="crest", vmin=1)

        ## save table for each PLO category
        dfi.export(df_style, f"{output_folder}/{PLO}_table.png")
        #dfi.export(df_t_style, f"{output_folder}/{PLO}_table_t.png", max_cols=-1)

def long_overview(plot_dfs):
    # loop over the different PLO categories
    for PLO in plot_dfs.keys():
        # create a 2 X 3 figure for Kunnskapar and 3 X 3 for F & GK
        fig, axs = plt.subplots(int(len(plot_dfs[PLO]["PLO"].unique())/3), 3, sharey=True, sharex=True)

        plt.tight_layout()

        # add the plots to the figure 1 by 1
        for LO, ax in zip(plot_dfs[PLO]["PLO"].unique(), axs.ravel()):
            # select the rows in the df corresponding to the learning outcome (LO)
            LO_df = plot_dfs[PLO][plot_dfs[PLO]["PLO"] == LO].copy()
            # add a column where the total coverage will be saved
            LO_df.loc[:,"Total coverage"] = 0

            # loop over all rows
            for i in LO_df.index:
                # for the first row, the total coverage == coverage
                if i == LO_df.index[0]:
                    LO_df.loc[i, "Total coverage"] = LO_df.loc[i, "coverage"]
                    level = LO_df.loc[i, "level"]

                # for all other rows, calculate the total coverage by adding the coverage of the row to the total coverage of the previous
                # row for each level independent. If the level changes, start with a total coverage of 0 again
                else:
                    if LO_df.loc[i, "level"] == level:
                        LO_df.loc[i,"Total coverage"] = LO_df.loc[i,"coverage"] + LO_df.loc[i_old,"Total coverage"]
                    else:
                        LO_df.loc[i, "Total coverage"] = LO_df.loc[i, "coverage"]
                        level = LO_df.loc[i, "level"]

                i_old = i

            sns.barplot(data=LO_df, x="semester", y="Total coverage", hue="level", palette=itemgetter(0,3,5)(sns.color_palette("viridis_r")), ax=ax).set_title(LO)
            ax.set(xlabel=None, ylabel=None)
        fig.supxlabel('Semester')
        fig.supylabel('Number of courses that cover PLO')

def main():
    # import data

    # for now, only import the finished mapping file
    path = "data/first_mapping/"
    semesters = "data/course_semesters.xlsx" # semester file can be found on the drive as well
    course_df = import_data(path, semesters, random_data="no")

    # specify input levels used in mapping
    levels = [1, 2, 3]

    ## create plots for longitudinal overview & course view
    views = ["semester", "course"]
    # indicate which semesters our courses should be visualized in the barplots
    # if selection = "all", all courses and semesters are considered
    selection = {"semester": [1, 2], "course":["PRPSYK100", "PRPSYK301A"]}

    for view in views:
        # calculate the PLO coverage for over all semesters or courses for each level
        coverage, rel_coverage, max_coverage = PLO_coverage(course_df, levels, view)
        # transform the PLO coverage table to a table that can be used as input of the bubble plot
        plot_dfs = transform_dfs(coverage, levels, view)

        # make a plot for each of the learning outcome categories
        bubble_plot(plot_dfs, view, level=1)
        # make a barplot for each learning outcome category
        bar_plot(plot_dfs, view, selection, course_df)

        # if we want the course overview, also create tables that have the maximum PLO coverage
        # this function saves the heatmap images of the pandas table in the specified output_folder
        if view == "course":
            table_view(max_coverage, levels, selection="all", output_folder="tables")

        # for the semester view, create longitudinal overview: barplot showing the increase in coverage
        if view == "semester":
            long_overview(plot_dfs)

if __name__ == '__main__':
    main()