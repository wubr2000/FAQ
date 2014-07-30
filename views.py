from flask import Flask
from flask import request
from flask import Flask, make_response
import os
import sqlite3
from flask import request, session, g, redirect, url_for, abort, \
     render_template, flash
import faq_clean as cleaned
from generateFAQ import*
import psycopg2
import cPickle


import os

app = Flask(__name__)
print 'server running'
#set questions per group
global questionsPerGroup
questionsPerGroup = 5
pathname = os.path.dirname(os.path.realpath(__file__))+"/"
filename = "statsLearningForum2.csv"

global sessionID
sessionID = 0

##### MODEL PARAMETERS #############
#n_features = 10000
n_top_words = 15
n_top_questions = 5
n_answers = 5
####################################


#@app.route('/t', methods=['GET', 'POST'])
def saveData (course_name="Medicine/HRP258/Statistics_in_Medicine",nQuestions=n_top_questions,algoValue=20):
    '''generateFAQ should be true when data needs to be saved to db '''
    #get session id
    #sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
    f = cleaned.pull_data(pathname, filename, course_name)
    all_questions, all_comments = cleaned.questions_comments(f)
    X, tfidf, n_topics = vectorize(all_comments)
    if not algoValue:
        algoValue = n_topics
    nmf, feature_names = nmf_model(X, tfidf, int(algoValue))
    faq, top_answers = Q_and_A(f, X, nmf, all_questions, n_top_questions, n_answers)
    histograms(nmf, tfidf, n_top_words)
    word_cloud(nmf, tfidf)
    flist = flaskList(nmf, faq, top_answers, n_top_questions, n_answers)

##    nmf = cPickle.load(open(pathname+'nmf.pkl','rb'))
##    tfidf = cPickle.load(open(pathname+'tfidf.pkl','rb'))
##    n_topics = cPickle.load(open(pathname+'n_topics.pkl','rb'))
##    feature_names = cPickle.load(open(pathname+'feature_names.pkl','rb'))
##    histograms(nmf, tfidf, n_top_words)
##    word_cloud(nmf, tfidf)
##    flist = cPickle.load(open(pathname+'flaskList.pkl','rb'))
    mainList = flist
    #mainList = [[[0, 62728, 'hello', ('a', 'b')]]]
    #add groups and questions to database
    positionNumber = 0
    groupID = 0
    count = 0
    for i in mainList:
        if count >= questionsPerGroup:
            count = 0
            groupID += 1
        #insert questions
        question = i[0].encode('ascii', 'ignore')
        alterDB("INSERT into questions(text,position,sessionid,groupid) VALUES(%s,%s,%s,%s);",[question,positionNumber,sessionID,groupID])
        positionNumber += 1
        count += 1
        #add answers to database
        answerList = i[1]
        answerPosition = 0
        for answer in answerList:
            answer = answer.encode('ascii', 'ignore')
           #get question id
            questionID = retrieveData("SELECT id FROM questions ORDER BY id DESC LIMIT 1",[])[0][0]
            alterDB("INSERT into answers(text,questionid,position)VALUES(%s,%s,%s);",[answer,questionID,answerPosition])
            answerPosition +=1

    #alterDB("INSERT into featuredwords (word,groupid,sessionid) VALUES(%s,%s,%s);",["hi",0,1])
    #save featured words
    string = ''
    for i in range(len(feature_names)):
       words = feature_names[i]
       for word in words:
            string += word + ','
       alterDB("INSERT into featuredwords (word,groupid,sessionid) VALUES(%s,%s,%s);",[string,i,sessionID])

    return n_topics

#@app.route('/test', methods=['GET', 'POST'])
def createUI(calltwo = False):
    #get session id
    #sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]

    if calltwo:
        sessID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
        questions = retrieveData("SELECT id, text, groupid FROM questions WHERE sessionid = %s ORDER BY position ASC",[sessID])
    else:
         questions = retrieveData("SELECT id, text, groupid FROM questions WHERE sessionid = %s ORDER BY position ASC",[sessionID])
    featuredWordList = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    #get questions
    masterList = []
    questionList = []
    count = 0
    groupList = []
    for i in questions:
        questionID = i[0]
        question = i[1]
        groupID = i[2]
        questionList.append([groupID,questionID,question,('a','b')])
        count += 1
        if count >= questionsPerGroup:
            groupList.append(questionList)
            questionList = []
            count = 0
    masterList.append(groupList)


    #return render_template('test.htm', debug = masterList)
    groupNumber = len(masterList)

    return [groupNumber,masterList,featuredWordList,sessionID]


def buildPNGList():
    pngList = []
    for i in range(25):
        #change path on different computer
        topic = "topic" + str(i) + ".png"
        path = "/static/" + topic
        pngList.append(path)
    return pngList

def createWordList():
    #create featured word list
    wordString = retrieveData("SELECT word FROM featuredwords WHERE sessionid = %s",[sessionID])[0][0]
    splitList = wordString.split(',')
    wordList = []
    subList = []
    subCount = 0
    for word in splitList:
        if subCount < 15:
            subList.append(word)
            subCount += 1
            continue
        wordList.append(subList)
        subCount = 0
        subList = []
    return wordList

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/faq', methods=['POST'])
def faq():
    groups = 0
    numberList = [1,2,3,4,5]
    
    #if generate faq buttton pressed, get new data
    generate = request.form.get('generateFAQ')
    if generate is not None:
       #create session id
       alterDB("INSERT into sessions(number) VALUES(1);",[])
       global sessionID
       sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]

       algoValue = request.form.get('algoValue')
       #save data
       n_topics_result  = saveData(course_name="Medicine/HRP258/Statistics_in_Medicine", algoValue = algoValue)
       if not algoValue:
           algoValue = n_topics_result
       data = createUI()
       groupNumber = data[0]
       questionList = data[1]
       featureNames = data[2]
       pngList = buildPNGList()
       #create featured word list
       wordList = createWordList()
       return render_template('generateFAQBruno.htm', groupNumber= groupNumber,questionList= questionList,pngList=pngList,featureNames = featureNames, n_topics=int(algoValue), postAddress = "/faq",wordList= wordList,debug = '', course = "Statistics in Medicine")
       #return render_template('test.htm', groupList= groupList,numberList= numberList,questionList= questionList,debug = data)
    #data saved from edit view and delete button
    updateFAQ = request.form.get('updateFAQ')
    if updateFAQ is not None:
       questionID = request.form['questionID']
       question = request.form['updateFAQ']
       #update to new question
       alterDB("UPDATE questions SET text = %s WHERE id = %s;",[question,questionID])
       #get answers for question
       answers = retrieveData("SELECT text, id FROM answers WHERE questionid = %s",[questionID])
    #update answers in db

       for i in answers:
           answer = i[0]
           answerID = i[1]
           #newAnswer = request.form[answer]
           newAnswer = request.form['tod']
           #problem here
           if newAnswer != answer:
               alterDB("UPDATE answers SET text = %s WHERE id = %s;",[newAnswer,answerID])

    data = createUI(calltwo= True)
    groupNumber = data[0]
    questionList = data[1]
    featureNames = data[2]
    pngList = buildPNGList()
    #return render_template('test.htm', debug = sessionID)
    return render_template('generateFAQBruno.htm', groupNumber= groupNumber,questionList= questionList,pngList=pngList,featureNames = featureNames, postAddress = "/faq", debug = '', course = "Statistics in Medicine")


@app.route('/', methods=['GET'])
def index():
    return render_template('generateFAQStart.html')
    

@app.route('/edit', methods=['GET', 'POST'])
def edit(questionList=[1],answersList=[],question=None):
    #postAddress
    postAddress = request.form.get('postAddress')
    if not postAddress:
        postAddress = "/faq"

    answerList = []
    if request.method == 'POST':
        questionID = request.form['questionID']
        question = request.form['question']
        answers = retrieveData("SELECT text FROM answers WHERE questionid = %s ORDER BY position ASC",[questionID])
        for answer in answers:
            answerList.append(answer[0])
        return render_template('edit.html', questionList=questionList,answerList=answerList,question = question, questionID=questionID,postAddress = postAddress, debug = '') #postaddress needs to be updated to tell where edit is coming from

    return render_template('edit.html',)

@app.route('/output', methods=['GET', 'POST'])
def output():
    questionList = []
    #sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
    questions = retrieveData("SELECT text,id FROM questions WHERE sessionid = %s ORDER BY position ASC",[sessionID])
    for i in questions:
        question = i[0]
        questionID = i[1]
        #get answers for question
        answers = retrieveData("SELECT text  FROM answers WHERE questionid = %s ORDER BY position ASC",[questionID])
        questionList.append([question,answers])
    return render_template('output.html',questionList = questionList)

@app.route('/statistical', methods=['GET', 'POST'])
def statistical():
     #create session id
     alterDB("INSERT into sessions(number) VALUES(1);",[])
     global sessionID
     sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
     algoValue = request.form.get('algoValue')
     n_topics_result = saveData(course_name = "HumanitiesScience/StatLearning/Winter2014", algoValue = algoValue)
     if not algoValue:
        algoValue = n_topics_result
     data = createUI()
     groupNumber = data[0]
     questionList = data[1]
     featureNames = data[2]
     pngList = buildPNGList()
     wordList = createWordList()
     return render_template('generateFAQBruno.htm', groupNumber= groupNumber,questionList= questionList,pngList=pngList,featureNames = featureNames, n_topics = int(algoValue), postAddress = "/statistical",wordList = wordList, debug = '', course = "Statistical Learning")

@app.route('/medicine', methods=['GET', 'POST'])
def medicine():
     #create session id
     alterDB("INSERT into sessions(number) VALUES(1);",[])
     global sessionID
     sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
     algoValue = request.form.get('algoValue')
     n_topics_result = saveData(course_name = "Medicine/HRP258/Statistics_in_Medicine", algoValue = algoValue)
     if not algoValue:
        algoValue = n_topics_result
     data = createUI()
     groupNumber = data[0]
     questionList = data[1]
     featureNames = data[2]
     pngList = buildPNGList()
     data = createUI()
     wordList = createWordList()
     return render_template('generateFAQBruno.htm', groupNumber= groupNumber,questionList= questionList,pngList=pngList,featureNames = featureNames, n_topics = int(algoValue), postAddress = "/medicine", wordList = wordList, debug = '', course = "Statistics in Medicine")


@app.route('/environmental', methods=['GET', 'POST'])
def environmental():
     #create session id
     alterDB("INSERT into sessions(number) VALUES(1);",[])
     global sessionID
     sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
     algoValue = request.form.get('algoValue')
     n_topics_result = saveData(course_name = "HumanitiesSciences/EP101/Environmental_Physiology", algoValue = algoValue)
     if not algoValue:
        algoValue = n_topics_result
     data = createUI()
     groupNumber = data[0]
     questionList = data[1]
     featureNames = data[2]
     pngList = buildPNGList()
     wordList = createWordList()
     return render_template('generateFAQBruno.htm', groupNumber= groupNumber,questionList= questionList,pngList=pngList,featureNames = featureNames, n_topics = int(algoValue), postAddress = "/environmental", wordList = wordList, debug = '', course = "Environmental Physiology")

@app.route('/engineering', methods=['GET', 'POST'])
def engineering():
     #create session id
     alterDB("INSERT into sessions(number) VALUES(1);",[])
     global sessionID
     sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
     algoValue = request.form.get('algoValue')
     n_topics_result = saveData(course_name = "Engineering/db/2014_1", algoValue = algoValue)
     if not algoValue:
        algoValue = n_topics_result
     data = createUI()
     groupNumber = data[0]
     questionList = data[1]
     featureNames = data[2]
     pngList = buildPNGList()
     wordList = createWordList()
     return render_template('generateFAQBruno.htm', groupNumber= groupNumber,questionList= questionList,pngList=pngList,featureNames = featureNames, n_topics = int(algoValue), postAddress = "/engineering",  wordList = createWordList(), debug = '', course = "Introduction to Database")

@app.route('/psych', methods=['GET', 'POST'])
def psych():
     #create session id
     alterDB("INSERT into sessions(number) VALUES(1);",[])
     global sessionID
     sessionID = retrieveData("SELECT id FROM sessions ORDER BY id DESC LIMIT 1",[])[0][0]
     algoValue = request.form.get('algoValue')
     n_topics_result = saveData(course_name = "EarthSciences/GP202/Spring2014", algoValue = algoValue)
     if not algoValue:
        algoValue = n_topics_result
     data = createUI()
     groupNumber = data[0]
     questionList = data[1]
     featureNames = data[2]
     pngList = buildPNGList()
     wordList = createWordList()
     return render_template('generateFAQBruno.htm', groupNumber= groupNumber,questionList= questionList,pngList=pngList,featureNames = featureNames, n_topics = int(algoValue), postAddress = "/psych",  wordList = createWordList(), debug = '', course = "Reservoir Geomechanics")


@app.route('/query', methods=['GET', 'POST'])
def query():
    query = request.form.get('queryI')
    if query:
        nmf = cPickle.load(open(pathname+'nmf.pkl','rb'))
        faq = cPickle.load(open(pathname+'FAQ.pkl','rb'))
        tfidf = cPickle.load(open(pathname+'tfidf.pkl','rb'))
        closest_topics, related_q = related_questions([query],nmf,faq,tfidf)
        return render_template('query.htm',closest_topics=closest_topics, related_q= related_q,debug = '')
    return render_template('query.htm',closest_topics=[], related_q=[],debug = '')


@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
    query = request.form.get('queryI')
    nmf = cPickle.load(open(pathname+'nmf.pkl','rb'))
    faq = cPickle.load(open(pathname+'FAQ.pkl','rb'))
    tfidf = cPickle.load(open(pathname+'tfidf.pkl','rb'))
    closest_topics, related_q = related_questions([query],nmf,faq,tfidf)
    return render_template('autocomplete.htm',closest_topics=closest_topics, related_q= related_q,debug = '')


def retrieveData(sqlCommand,argList):
    conn = psycopg2.connect("dbname=faq user=wubr2000 password='phantom'") 
    cur = conn.cursor()
    EX = cur.execute(sqlCommand,argList)
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result

def alterDB(sqlCommand,data):
    """don't forget second input (list)"""
    conn = psycopg2.connect("dbname=faq user=wubr2000  password='phantom'")
    cur = conn.cursor()
    EX = cur.execute(sqlCommand,data)
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=80)

