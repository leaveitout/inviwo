#*********************************************************************************
#
# Inviwo - Interactive Visualization Workshop
#
# Copyright (c) 2013-2015 Inviwo Foundation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
#*********************************************************************************

import io
import os
import time
import subprocess

from . error import *
from .. util import *

class RunSettings:
	def __init__(self, timeout = 15):
		self.timeout = timeout

class InviwoApp:
	def __init__(self, appPath, settings = RunSettings()):
		self.program = appPath
		self.settings = settings

	def runTest(self, test, report, output):
		outputdir = test.makeOutputDir(output)

		for workspace in test.getWorkspaces():

			starttime = time.time()
			report['outputdir'] = outputdir

			command = [self.program, 
						"-q",
						"-o", outputdir, 
						"-g", "screenshot.png",
						"-s", "UPN", 
						"-l", "log.txt",
						"-w", workspace]
			report['command'] = " ".join(command)

			report['timeout'] = False

			try:
				process = subprocess.Popen(
					command,
					cwd = os.path.dirname(self.program),
					stdout=subprocess.PIPE, 
					stderr=subprocess.PIPE,
					universal_newlines = True
				)
			except FileNotFoundError:
				raise MissingInivioAppError("Could not find inviwo app at: {}".format(self.program))

			try:
				report["output"], report["errors"] = process.communicate(timeout=self.settings.timeout)
			except subprocess.TimeoutExpired as e:
				report['timeout'] = True
				process.kill()
				report["output"], report["errors"] = process.communicate()
			
			report['log'] = outputdir + "/log.txt"
			report['screenshot'] = outputdir + "/screenshot.png"
			report['returncode'] = process.returncode
			report['elapsed_time'] = time.time() - starttime

			return report



		