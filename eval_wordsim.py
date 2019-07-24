import sys
import numpy.linalg as linalg
import numpy as np
import scipy.stats as stats

# Read word embedding file 
wordvecFile = sys.argv[1]
# print_writer_filename = sys.argv[2]
wordVecDict = {}
file = open(wordvecFile, 'r', encoding='utf-8')
# fprint = open(print_writer_filename, 'a', encoding='utf-8')
num = 0
for line in file:
    num += 1
    if num % 500 == 0:
        print("Reading the %d-th word" % num)

    items = line.strip().split()
    items = [item.strip('[],') for item in items]
    word = items[0]
    vec = list(map(float, items[1:]))
    if linalg.norm(vec) != 0:
        # Important!
        if word not in wordVecDict:
            wordVecDict[word] = vec / linalg.norm(vec)
        else:
            wordVecDict[word] = (vec / linalg.norm(vec) + wordVecDict[word]) / 2
file.close()
print('Word embeddings reading completes and total number of words is:', num)
# fprint.write('Word embeddings reading completes and total number of words is:' + str(num) + '\n')

# test on wordsim240 and wordsim297
wordSimType = ['240', '297']
for x in wordSimType:
    file = open('wordsim/filtered_wordsim' + x + '.txt', 'r', encoding='utf-8')
    testPairNum = 0
    skipPairNum = 0

    wordSimStd = []
    wordSimPre = []
    for line in file.readlines():
        word1, word2, valStr = line.strip().split()
        if (word1 in wordVecDict) and (word2 in wordVecDict):
            testPairNum += 1
            wordSimStd.append(float(valStr))
            wordVec1 = wordVecDict[word1]
            wordVec2 = wordVecDict[word2]
            cosSim = np.dot(wordVec1, wordVec2) / np.linalg.norm(wordVec1) / np.linalg.norm(wordVec2)
            wordSimPre.append(cosSim)
        else:
            skipPairNum += 1
            print('Skip:', word1, word2)
            # fprint.write('Skip: ' + word1 + '  ' + word2 + ' \n')
    # corrCoef = np.corrcoef(wordSimStd, wordSimPre)[0, 1]
    SpearCoef = stats.spearmanr(wordSimStd, wordSimPre).correlation
    print("WordSim-" + x + " Score:" + str(SpearCoef))
    print('TestPair:', testPairNum, 'SkipPair:', skipPairNum)
    # fprint.write("WordSim-" + x + " Score:" + str(SpearCoef) + '\n')
    # fprint.write('TestPair:' + str(testPairNum) + 'SkipPair:' + str(skipPairNum) + '\n')
    file.close()

# test on COS960
file = open('wordsim/COS960.txt', 'r', encoding='utf-8')
testPairNum = 0
skipPairNum = 0

wordSimStd = []
wordSimPre = []
for line in file.readlines():
    word1, word2, valStr = line.strip().split()
    if (word1 in wordVecDict) and (word2 in wordVecDict):
        testPairNum += 1
        wordSimStd.append(float(valStr))
        wordVec1 = wordVecDict[word1]
        wordVec2 = wordVecDict[word2]
        cosSim = np.dot(wordVec1, wordVec2) / np.linalg.norm(wordVec1) / np.linalg.norm(wordVec2)
        wordSimPre.append(cosSim)
    else:
        skipPairNum += 1
        print('Skip:', word1, word2)
        # fprint.write('Skip: ' + word1 + '  ' + word2 + ' \n')
# corrCoef = np.corrcoef(wordSimStd, wordSimPre)[0, 1]
SpearCoef = stats.spearmanr(wordSimStd, wordSimPre).correlation
print("WordSim960 Score:" + str(SpearCoef))
print('TestPair:', testPairNum, 'SkipPair:', skipPairNum)
# fprint.write("WordSim960 Score:" + str(SpearCoef) + '\n')
# fprint.write('TestPair:' + str(testPairNum) + 'SkipPair:' + str(skipPairNum) + '\n')
file.close()
# fprint.close()
