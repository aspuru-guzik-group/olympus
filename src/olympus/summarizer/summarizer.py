#!/usr/bin/env python

import json
import pandas as pd

from olympus.analyzer import Analyzer

#===============================================================================

class Summarizer():
	def __init__(self, campaigns=[], include_stats = True):
		''' class to summarize the results of an Olympus benchmark into digestible
		format --> latex table, csv, xlsx

		Args:
			campaigns (list): list of Olympus campaigns
			inc_analysis (bool): specifies whether or not to include statistics in the tables

		 '''
		self.campaigns = campaigns
		self.include_stats = include_stats
		if self.include_stats:
			self.analyzer = Analyzer(campaigns = campaigns)
			self.stats_map = {'mean':   self.analyzer.get_best_mean,
							  'median': self.analyzer.get_best_median,
							  'std':    self.analyzer.get_best_std,
							  'min':    self.analyzer.get_best_min,
							  'max':    self.analyzer.get_best_max
						}


	def _load_json(self):
		''' load json file containing the saved citation information '''
		self.dataset_bib = json.load(open('olympus/summarizer/dataset_bib.json', 'r'), strict = False)
		self.planner_bib  = json.load(open('olympus/summarizer/planner_bib.json', 'r'), strict = False)

	def _sort_campaigns(self, campaigns):
		''' convert a list of campaigns to a dictionary where the keys
		are the planner names and the values are sublists of campaigns all having the
		same planner.

		Args:
			campaigns (list): a list of olympus campaign objects
		Return:
			sorted_campaigns (dict): dictionary of campaign lists
		'''
		sorted_campaigns = dict()
		for campaign in campaigns:
			if not campaign.planner_kind in sorted_campaigns.keys():
				sorted_campaigns[campaign.planner_kind] = [campaign]
			else:
				sorted_campaigns[campaign.planner_kind].append(campaign)
		return sorted_campaigns

	def _get_stats(self, sorted_campaigns, stats_list, locs, precision = 1):
		''' generate statistics for the various campigns
			possible statistics:
				DEFAULT: mean
				OPTIONAL: median, std, min, max

		Args:
			sorted_campaigns (dict): dictionary of campaign lists
			stats_lists (list): list of the statistics to compute
			locs (array-like): array of locations at which to compute the stats
			precision (int): the number of decimals to include in the table
		Returns:
			stats_dict (dict): dictionary with planner stats in it
		'''
		stats_dict = dict()
		for planner_name, campaigns in sorted_campaigns.items():
			stats = dict()
			for stat in stats_list:
				if stat == 'mean':
					res = self.stats_map[stat](campaigns = campaigns,
												locs      = locs,
												ci_method = None,
												ci_size   = 100)
				else:
					res = self.stats_map[stat](campaigns = campaigns,
									   		   locs      = locs)
				if len(res) == 1:
					res = res[0]
					res = round(res, precision)
				stats[stat] = res
			stats_dict[planner_name] = stats
		return stats_dict

	def _write_bib(self, campaigns, filename  = 'summary.bib'):
		''' write all of the dataset and planner citations to a .bib file

		Args:
			campaigns (list): a list of campiagns from which the Bibtex entries are
			referenced
		Returns:
			None
		'''
		datasets = list(set(campaign['dataset_kind'] for campaign in campaigns))
		planners = list(set(campaign['planner_kind'] for campaign in campaigns))
		with open(filename, 'w') as f:
			f.write('%==== DATASETS ======================\n\n')
			for dataset in datasets:
				f.write(self.dataset_bib[dataset]['bibtex'] + '\n')
			f.write('\n%==== PLANNERS ======================\n\n')
			for planner in planners:
				f.write(self.planner_bib[planner]['bibtex'] + '\n')
			f.close()

	def _write_table_capton(self, dataset, model_kind, goal):
		'''
		Write a caption for the latex table

		Args:
			dataset (str): the name of the dataset in the campaign
			model_kind (str): the name of the model used in the emulator
			goal (str): the name of the optimization goal (maximize or minimize)
		Returns:
			caption (str): a string table caption
		'''
		caption = f'Benchmark results on the {dataset} dataset emulated using a {model_kind} with the goal {goal}.'
		return caption

	def to_latex(self, filename = 'summary.tex', stats_list = ['mean']):
		''' Write the summary to a LaTeX table, and write the citations
		to a .bib file

		Args:
			filename (str): the name of the .tex and .bib files you wish to save
			stats_list (list): list of statistics to include in the table

		Returns:
			None
		'''
		self._load_json()
		caption = self._write_table_capton(self.campaigns[0]['dataset_kind'],
										   self.campaigns[0]['model_kind'],
										   self.campaigns[0]['goal'])

		if self.include_stats:
			# compute the stats by default, the stats are computed at the final evaluation of the campaign
			sorted_campaigns = self._sort_campaigns(self.campaigns)
			stats_dict = self._get_stats(sorted_campaigns,
										 stats_list,
										 [self.campaigns[0].observations.get_values().shape[0]])

			df = pd.DataFrame(stats_dict)
			# replace the row names with the name and the citation
			columns = df.columns
			# rename_map = dict()
			# for column in columns:
			# 	rename_map[column] = column + ' /cite{' + ','.join(self.planner_bib[column]['cite_keys']) + '}'
			# df = df.rename(columns = rename_map)
			# switch the rows and columns
			df = df.transpose()
			latex_tab = df.to_latex(index   = True,
									caption = caption,
									label   = 'tab:benchmark_results')
		else:
			planners = list(set([c.planner_kind for c in self.campaigns]))
			df = pd.DataFrame({'planner': planners})
			latex_tab = df.to_latex(index   = False,
									caption = caption,
									label   = 'tab:planners')
			# write the table
			with open(filename, 'w') as f:
				f.write(latex_tab)
				f.close()

			self._write_bib(self.campaigns)


	def to_csv(self, filename = 'summary.csv', stats_list = ['mean']):
		'''Write the summary to a .csv file

		Args:
			filename (str): the name of the .tex and .bib files you wish to save
			stats_list (list): list of statistics to include in the table

		Returns:
			None
		'''
		# TODO: What will we do with the citations here: omit for now
		if self.include_stats:
			# compute the stats
			sorted_campaigns = self._sort_campaigns(self.campaigns)
			# by default, the stats are computed at the final evaluation of the campaign
			stats_dict = self._get_stats(sorted_campaigns,
										 stats_list,
										 [self.campaigns[0].observations.get_values().shape[0]])
			df = pd.DataFrame(stats_dict).transpose()
		else:
			planners = list(set([c.planner_kind for c in self.campaigns]))
			df = pd.DataFrame({'planner': planners})
		df.to_csv(filename)



	def to_excel(self, filename = 'summary.xlsx', stats_list = ['mean']):
		'''Write the summary to an .xlsx file

		Args:
			filename (str): the name of the .tex and .bib files you wish to save
			stats_list (list): list of statistics to include in the table

		Returns:
			None
		'''
		# TODO: What will we do with the citations here: omit for now
		if self.include_stats:
			# compute the stats by default, the stats are computed at the final evaluation of the campaign
			sorted_campaigns = self._sort_campaigns(self.campaigns)
			stats_dict = self._get_stats(sorted_campaigns,
										 stats_list,
										 [self.campaigns[0].observations.get_values().shape[0]])
			df = pd.DataFrame(stats_dict).transpose()
		else:
			planners = list(set([c.planner_kind for c in self.campaigns]))
			df = pd.DataFrame({'planner': planners})
		df.to_excel(filename)
