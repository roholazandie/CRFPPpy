import os
from subprocess import Popen, PIPE
import subprocess
CRFPProotDirectory = "/home/rohola/CRF++-0.58/"
filesRoot = "/home/rohola/CRF++-0.58/example/basenp/"
#subprocess.check_call(["cd",CRFPProotDirectory],shell=True)

learnScript = CRFPProotDirectory+"crf_learn"
c = "-c"
cValue = "10.0"
templateFile = filesRoot+"template"
trainFile = filesRoot+"train.data"
modelFile = filesRoot+"model"

c = "-c"
c_value = "3"
f = "-f"
f_value = "2"
p = "-p"
p_value = "1"

process = Popen([learnScript,c,cValue,f,f_value,p,p_value,templateFile,trainFile,modelFile], stdout=PIPE)
(output, err) = process.communicate()
exit_code = process.wait()

print(output)
