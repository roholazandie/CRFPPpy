'''
Created on Jun 30, 2015

@author: rohola zandie
'''
import codecs
import os
from subprocess import Popen, PIPE
import subprocess

class ConditionalRandomField():
    '''
    classdocs
    '''


    def __init__(self, regularization_algorithm = "CRF-L2", c = 1, cutoff_threshold = 1, number_threads = 1):
        self.regularization_algorithm = regularization_algorithm
        self.c = c
        self.cutoff_threshold = cutoff_threshold
        self.number_threads = number_threads
        self.CRFPProotDirectory = "/home/rohola/CRF++-0.58/"
        self.filesRoot = "/home/rohola/CRF++-0.58/example/basenp/"
        
    
    def fit(self,train_data):
        '''
        There are 4 major parameters to control the training condition

        -a CRF-L2 or CRF-L1:
        Changing the regularization algorithm.
        Default setting is L2. Generally speaking, L2 performs slightly better than L1,
        while the number of non-zero features in L1 is drastically smaller than that in L2.
        -c float: 
        With this option, you can change the hyper-parameter for the CRFs.
        With larger C value, CRF tends to overfit to the give training corpus.
        This parameter trades the balance between overfitting and underfitting.
        The results will significantly be influenced by this parameter.
        You can find an optimal value by using held-out data or more general model selection method such as cross validation.
        -f NUM:
        This parameter sets the cut-off threshold for the features.
        CRF++ uses the features that occurs no less than NUM times in the given training data.
        The default value is 1. When you apply CRF++ to large data, the number of unique features would amount to several millions.
        This option is useful in such cases.
        -p NUM:
        If the PC has multiple CPUs, you can make the training faster by using multi-threading.
        NUM is the number of threads.
        '''
        
        
        '''
        with codecs.open(self.filesRoot+"train.data","w") as fileWriter:
            for feature_vector in train_data:
                fileWriter.write("\t".join(feature_vector)+"\n")
        '''
        
        learnScript = self.CRFPProotDirectory+"crf_learn"
        a = "-a"
        a_value = self.regularization_algorithm
        c = "-c"
        c_value = str(self.c)
        f = "-f"
        f_value = str(self.cutoff_threshold)
        p = "-p"
        p_value = str(self.number_threads)
        self.templateFile = self.filesRoot+"template"
        self.trainFile = self.filesRoot+"train.data"
        self.modelFile = self.filesRoot+"model"
        output = ""
        
        try:
            if a_value == "MIRA":
                process = Popen([learnScript,c,c_value,a,a_value,f,f_value,p,p_value,self.templateFile,self.trainFile,self.modelFile], stdout=PIPE)
                (output, err) = process.communicate()
                print(err)
                exit_code = process.wait()
            else:
                process = Popen([learnScript,c,c_value,f,f_value,p,p_value,self.templateFile,self.trainFile,self.modelFile], stdout=PIPE)
                (output, err) = process.communicate()
                exit_code = process.wait()
                
        except:
            print("Can not run the code. Check the paths.")
        
        self._extract_training_result(output)
        
    
    def predict(self,test_data):
        '''
        with codecs.open("test.data","w") as fileWriter:
            for feature_vector in test_data:
                fileWriter.write("\t".join(feature_vector)+"\n")
        '''
        
        testScript = self.CRFPProotDirectory+"crf_test"
        testFile = self.filesRoot+"test.data"
        model = '-m'
        try:
            
            process = Popen([testScript,model,self.modelFile,testFile], stdout=PIPE)
            (output, err) = process.communicate()         
                
        except:
            print("Can not run the code. Check the paths.")
        
        predicted = []
        for line in output.split('\n'):
            predicted.append(line.split('\t')[-1])
        
        return predicted
    
    
    def prob(self,test_data):
        '''
        The -v option sets verbose level. default value is 0.
        By increasing the level, you can have an extra information from CRF++

        level 1 
        You can also have marginal probabilities for each tag (a kind of confidece measure for each output tag)
        and a conditional probably for the output (confidence measure for the entire output).
        '''
        
        
        '''
        with codecs.open("test.data","w") as fileWriter:
            for feature_vector in test_data:
                fileWriter.write("\t".join(feature_vector)+"\n")
        '''
        
        testScript = self.CRFPProotDirectory+"crf_test"
        testFile = self.filesRoot+"test.data"
        model = '-m'
        verboseLevel = '-v1'
        try:
            process = Popen([testScript,verboseLevel,model,self.modelFile,testFile], stdout=PIPE)
            (output, err) = process.communicate()         
                
        except:
            print("Can not run the code. Check the paths.")
        
        probabilities = []
        lines = output.split('\n')
        for line in lines[1:]:
            if not line:
                break
            tag = line.split('\t')[-1].split('/')[0]
            prob = line.split('\t')[-1].split('/')[1]
            probabilities.append((tag,prob))
        
        return probabilities
    
    def probAll(self,test_data):
        '''
        You can also have marginal probabilities for all other candidates.
        '''
        
        '''
        with codecs.open("test.data","w") as fileWriter:
            for feature_vector in test_data:
                fileWriter.write("\t".join(feature_vector)+"\n")
        '''
        
        testScript = self.CRFPProotDirectory+"crf_test"
        testFile = self.filesRoot+"test.data"
        model = '-m'
        verboseLevel = '-v2'
        try:
            process = Popen([testScript,verboseLevel,model,self.modelFile,testFile], stdout=PIPE)
            (output, err) = process.communicate()         
                
        except:
            print("Can not run the code. Check the paths.")
        
        lines = output.split('\n')
        lines = [line.split('\t') for line in lines]
        
        probabilities = []
        for line in lines[1:]:
            verboseProbs = {}
            for i in range(len(line)):
                if '/' in line[i]:
                    tag = line[i].split('/')[0]
                    prob = line[i].split('/')[1]
                    verboseProbs[tag]=prob
            probabilities.append(verboseProbs)
            
        
        return probabilities
    
    
    def prob_Nbest(self,test_data,n_best=3):
        '''
        With the -n option, you can obtain N-best results sorted by the conditional probability of CRF.
        With n-best output mode, CRF++ first gives one additional line like "# N prob",
        where N means that rank of the output starting from 0 and prob denotes the conditional probability for the output.

        Note that CRF++ sometimes discards enumerating N-best results if it cannot find candidates any more.
        This is the case when you give CRF++ a short sentence.

        CRF++ uses a combination of forward Viterbi and backward A* search.
        This combination yields the exact list of n-best results
        '''
        
        '''
        with codecs.open("test.data","w") as fileWriter:
            for feature_vector in test_data:
                fileWriter.write("\t".join(feature_vector)+"\n")
        '''
        
        testScript = self.CRFPProotDirectory+"crf_test"
        testFile = self.filesRoot+"test.data"
        model = '-m'
        n = '-n'
        n_value = str(n_best)
        try:
            process = Popen([testScript,n,n_value,model,self.modelFile,testFile], stdout=PIPE)
            (output, err) = process.communicate()         
                
        except:
            print("Can not run the code. Check the paths.")
        
        lines = output.split('\n')
        lines = [line.split('\t') for line in lines if line]
        
        best_results = []
        for line in lines:
            if line[0].startswith('#'):
                continue
            best_result = []
            line.reverse()
            for i in range(n_best-1):
                best_result.append(line[i])
            best_result.reverse()
            best_results.append(best_result)
            
        
        return best_results
    
    def _extract_training_result(self,output):
        '''
        iter: number of iterations processed
        terr: error rate with respect to tags. (# of error tags/# of all tag)
        serr: error rate with respect to sentences. (# of error sentences/# of all sentences)
        obj: current object value. When this value converges to a fixed point, CRF++ stops the iteration.
        diff: relative difference from the previous object value.
        '''
        lines = output.split('\n')
        
        self.NumberOfSentences = int(lines[7].split(':')[1])
        self.NumberOfFeatures = int(lines[8].split(':')[1])
        self.Freq = int(lines[9].split(':')[1])
        self.eta = float(lines[10].split(':')[1])
        self.C =  float(lines[11].split(':')[1])
        self.shrinkingSize = float(lines[12].split(':')[1])
        
        self.iters = []
        self.terrs = []
        self.serrs = []
        self.acts = []
        self.objs = []
        self.diffs = []
        for line in lines[13:]:
            if not line:
                break
            items = line.split(' ')
            self.iters.append(float(items[0].split('=')[1]))
            self.terrs.append(float(items[1].split('=')[1]))
            self.serrs.append(float(items[2].split('=')[1]))
            self.acts.append(float(items[3].split('=')[1]))
            self.objs.append(float(items[4].split('=')[1]))
            self.diffs.append(float(items[5].split('=')[1]))
        
        
        
        
        
        
        