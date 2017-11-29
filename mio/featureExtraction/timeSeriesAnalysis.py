# Copyright 2017  Fredrik Wrede, Prashant Singh, and Andreas Hellander
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Time Series Analysis Class

"""
# Imports
from featureExtractionBase import FeatureExtractionBase
from pandas import DataFrame
import tsfresh
import numpy as np

class TimeSeriesAnalysis(FeatureExtractionBase):

	def __init__(self, name, gillespy_model=None, columns=None):
		super(TimeSeriesAnalysis, self).__init__(name, gillespy_model, columns)
		
	def put(self, data):
		"""TODO"""
		nr_datapoints = len(self.features)
		for enum, datapoint in enumerate(data):
			enum += nr_datapoints
			df = DataFrame(data=map(lambda x: np.concatenate(([enum, 0], x)), datapoint),
				columns = self.columns)
			self.data = self.data.append(df)
			

	def delete_row(self):
		"""TODO"""

	def delete_column(self):
		"""TODO"""

	def generate(self):
		"""TODO"""

		non_computed = self.data.loc[self.data['computed'] == 0] #filter non-computed rows
		try:
			assert len(non_computed) > 0
		except AssertionError:
			print 'All datapoints in __.data has already been computed or data is missing.\
				Add new datapoints first.'
		else:
			#remove the 'computed' column  (required by tsfresh)
			non_computed = non_computed.drop(['computed'], axis=1)

			#compute the features using tsfresh
			f_subset = tsfresh.extract_features(non_computed, column_id = "index", 
				column_sort = "time", column_kind = None, column_value = None)
			self.features = self.features.append(f_subset)
			self.data.loc[self.data['computed'] == 0, 'computed'] = 1 #warning code redundancy 
