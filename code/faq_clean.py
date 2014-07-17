import pandas as pd
import numpy as np
import bs4 as bs4
import re, sys

# pathname = "/Users/wubr2000/Dropbox/Data Science Courses/Zipfian project Ideas/FAQ/FAQ/data/"
# filename = "statsLearningForum2.csv"
# coursename = "Medicine/HRP258/Statistics_in_Medicine"

def pull_data(pathname, filename, coursename):
    '''
    INPUT: filename, coursename
    OUTPUT: forum

    Takes in the course discussion forum dataset and
    returns forum dataframe containing discussions for a particular course
    '''
    df = pd.read_csv(pathname+filename, quoting = 2, header = 0, 
                    escapechar = "\\", encoding = "ISO-8859-1")

    forum = df[df.course_display_name == coursename]

    #Take out quotes in column names
    forum.columns = [c.replace("'","") for c in forum.columns]

    return forum

def mrclean(text):
    '''
    INPUT: text
    OUTPUT: cleaned text
    
    Removes all untranslatable unicode
    '''
    c = ['\xe2','\x80','\x93','\xc2','\xa0','\n', '\r', '\u2019', '\n\n', '\x96', '\xed',
         'zipredac', 'phoneredac', 'nameRedac_anon_screen_name_redacted']  
    return re.sub('[%s]' % ''.join(c), '', text)

def soupText(eText):
    '''
    INPUT: eText
    OUTPUT: text after removing HTML tags using BeautifulSoup

    Explicitly calls utf-8, replaces all double quotes with single and
    calls on helper function mrclean
    '''
    eText = str(eText.encode("utf-8").replace("'",'"') )
    soup = bs4.BeautifulSoup(eText)
    cleanText = soup.findAll(text=True)  #.get_text()
    return mrclean(''.join(cleanText))

def questions_comments(forum):
    '''
    INPUT: forum (an already cleaned forum dataframe)
    OUTPUT: all_questions, all_comments

    Explicitly calls utf-8, replaces all double quotes with single and
    calls on helper function mrclean
    '''

    concat_comments = forum[forum.type=="Comment"].sort("comment_thread_id").groupby("comment_thread_id").body.sum()

    all_comments=[]
    for c in concat_comments:
        #unicode(c)
        soupText(c)
        all_comments.append(c)

    all_questions = []
    for i in concat_comments.index:
        q = forum.body[forum.forum_post_id == i]
        all_questions.append(q.iloc[0])

    return all_questions, all_comments




