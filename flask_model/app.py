from flask import Flask, request
from flask import json
from flask import jsonify
from flask_cors import CORS
from konlpy.tag import Okt
from konlpy.tag import Kkma
from keras import models
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

import pandas as pd
import numpy as np
import jpype
import re
import os
import time

app = Flask (__name__)
CORS(app)

cog_error=''
user_data = ['','']
finish=False
finish_=False

# --------------------- BiLSTM -----------------------

# KoNLPy 형태소분석기 설정
tagger = Okt()
# 사전 정의
PAD = "<PADDING>"
PAD_INDEX = 0
OOV = "<OOV>"
OOV_INDEX = 1

# 형태소분석 함수
def pos_tag(sentences):
    
    # 문장 품사 변수 초기화
    sentences_pos = []
    
    # 인풋이 리스트면
    if isinstance(sentences,list):
    # 모든 문장 반복
        for sentence in sentences:
            # [\"':;~()] 특수기호 제거
            sentence = re.sub("[ㄱ-ㅎㅏ-ㅣ-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]", " ", sentence)
            
            
            # 배열인 형태소분석의 출력을 띄어쓰기로 구분하여 붙임
            sentence = " ".join(tagger.morphs(sentence))
            sentences_pos.append(sentence)
            
    # str이면        
    elif isinstance(sentences, str):
        sentences=re.sub("[ㄱ-ㅎㅏ-ㅣ-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]", " ", sentences)
        sentences_pos= " ".join(tagger.morphs(sentences))
        
    return sentences_pos

# 정수 인코딩 및 패딩
def convert_text_to_index_for_classification(sentences, vocabulary): 
    sentences_index = []
    if isinstance(sentences,list):
        # 모든 문장에 대해서 반복
        for sentence in sentences:
            sentence_index = []

            # 문장의 단어들을 띄어쓰기로 분리
            for word in sentence.split():
                if vocabulary.get(word) is not None:
                    # 사전에 있는 단어면 해당 인덱스를 추가
                    sentence_index.extend([vocabulary[word]])
                else:
                    # 사전에 없는 단어면 OOV 인덱스를 추가
                    sentence_index.extend([vocabulary[OOV]])

            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]

            # 최대 길이에 없는 공간은 패딩 인덱스로 채움
            sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]

            # 문장의 인덱스 배열을 추가
            sentences_index.append(sentence_index)

    elif isinstance(sentences, str):
        sentence_index = []
        for word in sentences.split():
            if vocabulary.get(word) is not None:
                # 사전에 있는 단어면 해당 인덱스를 추가
                sentence_index.extend([vocabulary[word]])
            else:
                # 사전에 없는 단어면 OOV 인덱스를 추가
                sentence_index.extend([vocabulary[OOV]])

        if len(sentence_index) > max_sequences:
            sentence_index = sentence_index[:max_sequences]

        # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]

        sentences_index.append(sentence_index)
    return sentences_index


# 파일 불러오기
df_main = pd.read_csv('sentences.csv',encoding='utf-8')
a=df_main['Emotion'].unique()
category =list(a)
CATEGORY = len(category)

# 분노 0,슬픔 1, 중립 2, 행복 3,
# 카테고리 인덱스

category_to_index = {word: index for index, word in enumerate(category)}

index_to_category = {index: word for index, word in enumerate(category)}

words = []
ori_sentence =[]
# 데이터 프레임 list화
for i in range(len(df_main)):
    tmp =[]
    tmp.append(str(i+1))
    tmp.append(df_main.iloc[i].Sentence)
    tmp.append(df_main.iloc[i].Emotion)
    ori_sentence.append(tmp)
    
# 문장 형태소 분석 및 전처리
sente=[]
for i in ori_sentence:
    sente.append(i[1])

senten = pos_tag(sente)

# 단어들의 배열 생성
for sentence in senten:
    for word in sentence.split():
        words.append(word)

# 길이가 0인 단어는 삭제
words = [word for word in words if len(word) > 0]

# 중복된 단어 삭제
words = list(set(words))

# 제일 앞에 태그 단어 삽입
words[:0] = [PAD, OOV]

VOCAB_SIZE = len(words)
# print(VOCAB_SIZE)

# 문장 길이 확인
max_length = max(len(l) for l in senten)
avg_length = sum(map(len, senten))/len(senten)
# print('리뷰의 최대 길이 : {}'.format(max_length))
# print('리뷰의 평균 길이 : {}'.format(avg_length))
max_sequences= int(avg_length) + 15

# 단어와 인덱스의 딕셔너리 생성
word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}

# 모델 불러오기
load_c_model = models.load_model('main_lstm_cl.h5')

# 모델에 사용할 함수 정의
    
def predict_emotion(text):
    pre_text = pos_tag(text)
    pre_x=convert_text_to_index_for_classification(pre_text, word_to_index)
    result=np.argmax(load_c_model.predict(np.asarray(pre_x).reshape(1,max_sequences)))
    return index_to_category[result]

# --------------------- Doc 2 Vec -----------------------

# 파일 불러오기
df = pd.read_csv('ChatbotData.csv')
df_Q=df['Q']
df_A=df['A']

# Series -> list
list_Q_A=[]
for i in range(len(df_Q)):
    temp=[]
    temp.append(str(i+1))
    temp.append(df_Q.iloc[i])
    temp.append(df_A.iloc[i])
    
    list_Q_A.append(temp)

# 형태소 분석
kkma = Kkma()
filter_kkma = ['NNG', 'NNP','OL','VA','VV','VXV']

def tokenizer_kkma(doc):
#     jpype.attachThreadToJVM()       
    token_doc = ["/".join(word) for word in kkma.pos(doc)]
    return token_doc

def tokenize_kkma_noun_verb(doc):
#     jpype.attachThreadToJVM()
    token_doc = ["/".join(word) for word in kkma.pos(doc) if word[1] in filter_kkma]
    return token_doc

# token화 후 인덱싱 및 문서 태깅
token_Q_A = [(tokenizer_kkma(row[1]), row[0]) for row in list_Q_A]
tagged_Q_A = [TaggedDocument(d,[c]) for d,c in token_Q_A]

# 모델 만들기
import multiprocessing
# 내 컴에 있는 cpu 갯수 cores 에 저장
cores = multiprocessing.cpu_count()
# vector_size : 임베딩할 벡터 차원
# negaive : negative sampling
d2v_Q_A = doc2vec.Doc2Vec(
    vector_size = 100,
    alpha = 0.025,
    min_alpha = 0.025,
    hs = 1,
    negative = 0,
    dm = 0,
    dbow_words = 1,
    min_count = 1,
    workers = cores,
    seed = 0
)

# 단어 사전 만들기
d2v_Q_A.build_vocab(tagged_Q_A)

# 모델 학습
for epoch in range(4):
    print(f'{epoch} 회 학습 중...............')
    d2v_Q_A.train(tagged_Q_A,
                    total_examples = d2v_Q_A.corpus_count,
                    epochs = d2v_Q_A.epochs)
    d2v_Q_A.alpha -=0.0025
    d2v_Q_A.min_alpha = d2v_Q_A.min_alpha

# 모델 예측

def doc2_answer(input_question):
    token_test = tokenizer_kkma(input_question)
    predict_vector = d2v_Q_A.infer_vector(token_test)
    result = d2v_Q_A.docvecs.most_similar([predict_vector],topn=1)
    return list_Q_A[int(result[0][0])-1][2]


@app.route('/generateQ',methods=['POST'])
def generate_Question():

    # 외부 변수 정의
    global user_data # 상황, 자동적 사고, 감정을 담은 배열
    global finish # 마지막인지 확인
    global finish_ # 마무리인지 확인
    global cog_error # 인지적 오류

    #json 데이터를 받아옴
    
    user_A = request.form['text1']
    user_name = request.form['user']
    index = int(request.form['idx'])
    user_cnt=request.form['user_count']

    # 인지적 오류 A 리스트 인지적오류 : [설명,대답]
    cognitive_error={
        '사고의 확대와 축소':['자신의 좋지 않은 면은 확대하고 좋은 면은 축소하는 경우',
        f'너무 자신을 과소평가하는 것이 아닌지 생각해보세요. {user_name}님은 이미 충분히 잘하고 있는걸요!'],
        '':['','']
    }

    # 챗봇 고정 질문 리스트
    bot_a_list = {
        '고정':['왜 그런 생각이 드셨는지 설명해 주실 수 있나요?',
                f'최근에 {user_name}님이 본인을 그렇게 느낄만한 일들이 있었나요?',#상황
                '그런 상황에서 스쳐지나간 생각이 있으신가요?',# 자동적 사고
                f'그 때 느낀 감정은 어땠나요?', # 감정
                f"그럼 '{user_data[1]}' 라는 말은 본인에게 무엇을 '의미'하나요?"],
        '부정':['그렇다면 그것을 개선하기 위해서는 어떻게 해야 하나요?',
                '그렇게 되지 않는다면 최악의 경우는 어떻게 될까요?'],#핵심신념
        '긍정':'그렇게 생각하는 이유가 있다면 어떤것이죠?', #핵심신념
        '마무리' : ["그렇군요.. 지금까지 성실히 답해주셔서 감사해요😌",f"오늘 {user_cnt}회기 상담은 여기서 마무리 하겠습니다. 어때요! {user_name}님 본인의 생각에 대해 깊게 생각해보게 된 것 같나요?!"],
        '종료':[f'저는 {user_name}님을 알 수 있는 좋은 시간이 되었어요!. 오늘 대화 중 "{user_data[1]}" 라고 말씀하셨는데 제가 생각하기에는 그건 인지적 오류 중 "{cog_error}"에 해당하는 것 같아요. 이는 {cognitive_error[cog_error][0]}에 . {cognitive_error[cog_error][1]}',
                f'저는 {user_name}님을 알 수 있는 좋은 시간이 되었어요!. 오늘 대화 중 "{user_data[1]}" 라고 말씀하셨는데 이렇게 생각 하신 것에 과도한 일반화, 흑백논리 등과 같은 인지적 오류가 있을 수 있습니다! 평소 스처가는 생각에 대해 다시 한번 깊게 생각을 해보셨으면 해요!',
                f'앞서 "{user_data[0]}"라는 상황일 때 "{user_data[1]}"라는 생각처럼 주어진 상황 속에 즉각적으로 느끼는 생각을 말합니다. 인지행동치료는 {user_name}님의 🔥적극적인 참여🔥가 필요해요. 제가 방금 상담 결과를 바탕으로 "자동적 사고 기록지"를 한 줄 적었으니 참고해서 다음 상담 때까지 5개를 꼬옥~!🤙 작성해주세요~😉'],
        '끝내기' : '끝내시겠다니 알겠어요..😥 언제라도 제가 필요하면 찾아와주세요~ 👋'
                }
    # 감정에 따른 리액션 A 리스트
    bot_a_emotion={'행복':'와 너무 좋으셨겠어요!! 기쁨은 나누면 배가 된데요!! 제가 제곱으로 늘려드릴께요!!!',
                '분노':'대따대따 힘드셨겠다.. 어떻게 버티셨어요😥',
                '중립':'아.. 그러셨구나 충분히 그럴 수 있죠!',
                '슬픔':'마음아프시겠어요 ..😢 슬픔은 나누면 나눠진다고 했어요..!',
                '공감':f'맞아요 제 생각도 그래요. {user_name}님은 충분히 노력하셨어요.'} 
    bot_Q_list=[]
    
    # 유저가 원하면 종료
    if user_A == '끝내기':
        bot_Q_list.append(bot_a_list['끝내기'])
        response={
            'bot_Q':bot_Q_list,
            'end_talk': 'end_talk'
        }
        return jsonify(response)
    elif index == -1:
        bot_Q_list.append('')
    # 마지막 전 이면
    elif index < len(bot_a_list['고정']):
        
        # 상황 질문 뒤 사용자의 답변 상황에 저장 및 감정에 맞는 리액션 전달
        if index==2 : 
            user_data.append(user_A)
            emotion = predict_emotion(user_A)
            bot_Q_list.append(bot_a_emotion[emotion])
            bot_Q_list.append(bot_a_list['고정'][index])
            del user_data[0]

        # 자동적 사고 질문 뒤 사용자의 답변 자사에 저장
        elif index==3 : 
            user_data.append(user_A)
            del user_data[0]
            bot_Q_list.append(f'방금 말한 "{user_data[1]}" 와 같은 생각을 했을 때 느낀 "감정"은 어떤가요?')

        # 감정 질문 뒤 사용자의 답변 감정에 저장 및 인지적 오류 저장
        elif index==4 : 
            user_data.append(user_A)
            cog_error = doc2_answer(user_data[1])
            print('doc2vec 결과 : ',doc2_answer(user_data[1]))

            if cog_error in cognitive_error:
                ans = cog_error+'('+ cognitive_error[cog_error][0]+')'
                user_data.append(ans)
            else :
                user_data.append('')
                cog_error = ''
            bot_Q_list.append(bot_a_list['고정'][index])

        else :
            bot_Q_list.append(bot_a_list['고정'][index])

    # 마지막 고정 멘트시 유저 감정 분석
    elif index == 5 :
        emotion=predict_emotion(user_A)
        if emotion == '긍정':
            bot_Q_list.append(bot_a_list['긍정'])
            finish_ = True
        else :
            bot_Q_list.append(bot_a_list['부정'][0])

    # 끝났을 때       
    elif finish :
        if cog_error == '':
            bot_Q_list.append(bot_a_list['종료'][1])
            bot_Q_list.append(bot_a_list['종료'][2])
        else : 
            bot_Q_list.append(bot_a_list['종료'][0])
            bot_Q_list.append(bot_a_list['종료'][2])
        
        response={
            'bot_Q':bot_Q_list,
            # user_data 상황, 자사, 감정, 인지적오류, 핵심신념 순서
            'user_data':user_data,
            'talk_finish':'finish'
        }
        user_data = ['','']
        cog_error=''
        finish = False
        finish_ = False
        time.sleep(3)
        
        return jsonify(response)

    elif finish_:
        user_data.append(user_A)
        bot_Q_list.append(bot_a_list['마무리'][0])
        bot_Q_list.append(bot_a_list['마무리'][1])
        finish = True

    # 부정일 때
    else :
        bot_Q_list.append(bot_a_list['부정'][1])
        finish_=True
      
        



    # 질문 3초 지연
    time.sleep(3)
    
    response={'bot_Q':bot_Q_list}

    return jsonify(response)


if __name__ == "__main__":
    print("I'm ready!")
    app.run(host='192.168.0.163', port='5001', debug=True)