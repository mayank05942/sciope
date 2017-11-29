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
Feature Extraction Base Class
"""
#import
from pandas import DataFrame
import numpy as np
import gillespy

class FeatureExtractionBase(object):

	def __init__(self, name, gillespy_model=None, columns=None):
		self.name = name
		try:
			if gillespy_model is not None:
				assert type(gillespy_model) == gillespy.gillespy.Model,\
					"gillespy_model needs to be of type gillespy.gillespy.Model: \
						%r was given" % type(gillespy_model)

				self.columns = np.concatenate((['index', 'computed', 'time'],
					gillespy_model.listOfSpecies.keys()))

			elif columns is None:
				print "columns need to be defined if not using gillespy_model"
			else:
				assert type(columns) is np.ndarray  or type(columns) is list,\
					"columns need to be of type list or numpy.ndarray: %r was given" \
						% type(columns)
				assert len(columns) > 0, "columns need to be of 1D shape, for example ['A','B']"
				self.columns = np.concatenate((['index', 'computed', 'time'], columns))

			self.data = DataFrame(columns = self.columns)
			self.features = DataFrame()

		except ValueError:
			print "columns need to be of 1D shape, for example ['A','B']"
			raise

	def put(self, data):
		"""Abstract method to put data"""
		raise NotImplementedError("Subclass must implement abstract method")

	def delete_row(self):
		"""Abstract method to delete rows in all or one data container"""
		raise NotImplementedError("Subclass must implement abstract method")

	def delete_column(self):
		"""Abstract method to delete column in either data container"""
		raise NotImplementedError("Subclass must implement abstract method")

	def generate(self):
		"""Abstract method to generate features"""
		raise NotImplementedError("Subclass must implement abstract method")

