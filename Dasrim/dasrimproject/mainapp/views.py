from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib import auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from dasrimproject.settings import BASE_DIR
from .models import Account, MyDiagnosis, Record
from konlpy.tag import Okt
from konlpy.tag import Kkma

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

pk_value=0
# Create your views here.


def record(request, user_pk):
    Users = User.objects.get(pk=user_pk)

    if request.POST:
        myrecord = Record(record_user=Users)
        myrecord.situation=request.POST['situation']
        myrecord.feel=request.POST['feel']
        myrecord.automatic=request.POST['automatic']
        myrecord.writer=request.POST['writer']
        myrecord.save()
    
    records = Record.objects.filter(record_user=Users)
    cnt = records.count()
    print(cnt)
    record_elly = Record.objects.filter(record_user=Users, writer='elly')
    
    context = {
            'user': Users,
            'records': records,
            'cnt': cnt,
            'record_elly' : record_elly
        }
    return render(request, 'record.html', context)


def service(request, user_pk):
    Users = User.objects.get(pk=user_pk)
    que = MyDiagnosis.objects.filter(diagnosis=Users)

    if que.count() < 1:   # 데이터x
        datas = 0
    else:                 # 데이터o
        datas = 1

    context = {
            'user': Users,
            'data': datas,
        }
    return render(request, 'service.html', context)

@csrf_exempt
def consult(request, user_pk):
    User_ = User.objects.get(pk=user_pk)
    Users = Account.objects.get(user=User_)
    global pk_value
    user_data=[]
    start_ment=[
        # "'지금의 나' 그림을 그리느라 고생하셨어요! 이제부턴 저 🐘엘리🐘와 함께 상담을 시작해봐요! 언제라도 끝내기🔚를 입력해 상담을 종료할 수 있어요!",
        f'시작하기 전에! 인지 행동 치료에서 중요한 것은 {Users.nickname}님의 마음속 깊은 곳에 있는 생각을 알아내야 해요! 그리고 그걸 재구조화해야 해요!',
        f'앞으로 8~10개의 질문을 통해서 {Users.nickname}님의 인지 구조를 파악해볼게요! 질문에 천천히 생각해보시고 답해주세요😆, 그럼 시작해볼까요?',
        '오늘 하루 어땠나요?'
                ]

        
    print('first pass')
    myrecord = Record()
    myrecord.record_user = request.user
    myrecord.count=1
    myrecord.save()
    print('myrecord.pk:',myrecord.pk)
    pk_value = myrecord.pk

        
    
        
    if request.POST.getlist('user_data[]') != [] :
        
        # user_data 상황, 자사, 감정, 인지적오류, 핵심신념 순서
        # print('request : ',request.POST)
        print('second pass')
        myrecord = Record.objects.filter(record_user=User_, pk=pk_value)
        # 받아온 데이터
        print('myrecord:',myrecord)

        user_data = request.POST.getlist('user_data[]')
        print('user_data : ',user_data)

        myrecord.update(situation=user_data[0],
        automatic=user_data[1],
        feel=user_data[2],
        cognitive_error=user_data[3],
        core_belief=user_data[4],
        writer='elly')

    context = {
            'user': User_,
            'users': Users,
            'myrecord':myrecord,
            'ment': start_ment[:-1],
            'ment_final': start_ment[-1]
        }
        # 유저 데이터 DB에 저장
    return render(request, 'consult.html', context)    


def explain(request, user_pk):
    Users = User.objects.get(pk=user_pk)

    context = {
        'user': Users,
    }
    return render(request, 'explain.html', context)


def result(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    # que = MyDiagnosis.objects.get(diagnosis=Users)

    if request.POST:
        mydia = MyDiagnosis.objects.get(pk=dia_pk)
        mydia.feel = request.POST['feeling']
        mydia.save()
        sys.stdout.flush()
        
        summary = mydia.summary
        self_select = mydia.self_select
        feel = mydia.feel

        # ------------------------------ 정서내용 모델 ------------------------------

        file_path = os.path.join(BASE_DIR, 'mainapp/sentiment.csv')
        sentiment = pd.read_csv(file_path)

        POS = sentiment['value'] == 'POS'
        NEG = sentiment['value'] == 'NEG'
        is_POS = sentiment[POS]
        is_NEG = sentiment[NEG]
        POS_words = is_POS['word'].values.tolist()
        NEG_words = is_NEG['word'].values.tolist()

        kkma = Kkma()
        morphs = []
        pos = kkma.pos(summary) 
        # print('kkma형태소 분석 :  ', pos)
        for i in pos: # ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']
            if i[1] not in ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']:
                morphs.append(i[0])
            else:
                pass
        total = len(morphs)
        # print('나눠진 형태소 :   ', total)

        POS_score = 0
        count = 0
        match_list = []
        for i in range(0,total):
            if morphs[i] in POS_words:
                POS_score += 1
                count += 1
                match_list.append(morphs[i])
            elif morphs[i] in NEG_words:
                POS_score += -1
                count += 1
                match_list.append(morphs[i])
            else:
                pass
        # print('POS_score :   ',POS_score)
        # print(total, count)
        # print('match_list :  ', match_list)

        file_path = os.path.join(BASE_DIR, 'mainapp/emotion_dic.xlsx')
        emotion_dic = pd.read_excel(file_path)
        score_1 = emotion_dic['score'] == 1
        words_1 = emotion_dic[score_1]['word'].values.tolist()
        score_2 = emotion_dic['score'] == 2
        words_2 = emotion_dic[score_2]['word'].values.tolist()
        score_4 = emotion_dic['score'] == 4
        words_4 = emotion_dic[score_4]['word'].values.tolist()
        score_5 = emotion_dic['score'] == 5
        words_5 = emotion_dic[score_5]['word'].values.tolist()
        # print('words_5 :  ',words_5)

        okt = Okt()
        pos = okt.pos(summary, stem=True)
        morphs = []
        for i in range(0, len(pos)):
            morphs.append(pos[i][0])
        # print('morphs :  ',morphs)

        if POS_score>0:
            count_4 = 0
            count_5 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_4:
                    count_4 += 1
                elif morphs[i] in words_5:
                    count_5 += 1
                else:
                    pass
            # print('긍 : ', count_4, count_5)
            if count_4 < count_5:
                score_1 = 5
            else:
                score_1 = 4

        elif POS_score<0:
            count_1 = 0
            count_2 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_1:
                    count_1 += 1
                elif morphs[i] in words_2:
                    count_2 += 1
                else:
                    pass
            # print('부 : ', count_1, count_2)
            if count_2 < count_1:
                score_1 = 1
            else:
                score_1 = 2     
        else:
            score_1 = 3  

        print("정서상 점수: ", score_1)


        # ------------------------------ 자아상 모델 ------------------------------
        # 1~4: 1점 5~8: 2점 9~12: 4점 13~16: 5점
        
        # print(self_select)
        counts = [0,0,0,0,0]
        select_list1 = ['1','2','3','4']
        select_list2 = ['5','6','7','8']
        select_list4 = ['9','10','11','12']
        select_list5=['13','14','15','16']

        import re

        n = re.sub('[\[\]\']', '', self_select)
        self_select_list = n.replace(' ','').split(",")
        print('self_select_list : ',self_select_list)

        # 자아상에 입력o
        if mydia.self_image:
            for num in self_select_list:
                # print(num)
                if num in select_list1:
                    counts[0]+=1
                elif num in select_list2:
                    counts[1]+=1
                elif num in select_list4:
                    counts[3]+=1
                elif num in select_list5:
                    counts[4]+=1
            # print('self_select : ', self_select)
            print('counts : ', counts)

            # 최댓값을 가지는 인덱스 모두 찾기
            max_counts = []
            max_count = max(counts)
            for i, j in enumerate(counts):
                if j == max_count:
                    max_counts.append(i)
            print('max_count : ', max_count)
            print('max_counts : ',max_counts)

            # 점수내기
            if max_counts[0] == 0:
                score_2 = 1
                print('자아상점수: ', score_2)
            elif max_counts[0] == 1:
                score_2 = 2
                print('자아상점수: ', score_2)
            elif max_counts[0] == 3:
                if len(max_counts) != 1:
                    score_2 = 5
                    print('자아점수: ', score_2)
                else:
                    score_2 = 4
                    print('자아점수: ', score_2)
            else:
                score_2 = 5
                print('자아점수: ', score_2)

        # 자아상에 아무것도 입력x   
        else:
            # counts[2] += 1
            score_2 = 3
            print('자아상점수: ', score_2)
        

        # ------------------------------ 자기보고 모델 ------------------------------
        score_3 = int(feel)
        print('자기보고점수: ', score_3)

        # ------------------------------ 총합점수 모델 ------------------------------
        result = (score_1 + score_2 + score_3) / 3.0
        print('@@@@@@@@@@@@@@@@result: ',result)
        total_result = round(result, 2)
        print('총합우울점수: ', total_result)

        if float(result)<2:
            level = 1
        elif float(result)>=2 and float(result)<3:
            level = 2
        else:
            level = 3

        # db에 점수저장
        mydia.score1 = score_1
        mydia.score2 = score_2
        mydia.score3 = score_3
        mydia.total_score = total_result
        mydia.level = level
        mydia.save()

        context = {
            'user': Users,
            'data': mydia,
            'score_1': score_1,
            'score_2': score_2,
            'score_3': score_3,
            'total_score': total_result,
            'level' : level
        }
        
    return render(request, 'result.html', context)

def home(request):
    return render(request, 'home.html')

def que3(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    # que = MyDiagnosis.objects.get(diagnosis=Users)
    # mydias = MyDiagnosis.objects.filter(diagnosis=Users, pk=dia_pk)

    if request.POST:
        mydia = MyDiagnosis.objects.get(pk=dia_pk)
        # mydia.update(
        #     self_image = request.POST['symbol'],
        #     self_select = request.POST.getlist('symbol_emo')
        # )

        mydia.self_image = request.POST['symbol']
        mydia.self_select = request.POST.getlist('symbol_emo')
        mydia.save()

        context = {
            'user': Users,
            'data': mydia
        }
        return render(request, 'que3.html', context)

    context = {
    'user': Users,
    'data': mydia
    }

    return render(request, 'que3.html', context)

def que2(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    # que = MyDiagnosis.objects.get(diagnosis=Users)
    # mydias = MyDiagnosis.objects.filter(diagnosis=Users)

    # print(que)
    # print(que.diagnosis)
    if request.POST:
        mydia = MyDiagnosis.objects.get(pk=dia_pk)
        # mydia.update(
        #     summary = request.POST['summary']
        # )
        mydia.summary = request.POST['summary']
        mydia.save()

        context = {
            'user': Users,
            'data': mydia
        }
        return render(request, 'que2.html', context)

    context = {
        'user': Users,
        'data': mydia
    }

    return render(request, 'que2.html', context)

def que1(request, user_pk):
    Users = User.objects.get(pk=user_pk)
    # que = MyDiagnosis.objects.get(diagnosis=Users)
    
    if request.POST:
        mydia = MyDiagnosis()
        mydia.diagnosis = request.user
        mydia.image = request.FILES['images']
        mydia.save()
        # update_dia = MyDiagnosis.objects.filter(diagnosis=Users)
        # if len(update_dia) > 0:
        #     que = MyDiagnosis.objects.get(diagnosis=Users)
        #     print('update!@@@@@@@@@@')
        #     que.image = request.FILES['images']
        #     que.save()

        # else:
        #     print('no update@@@@@@@@@@@@')
        #     mydia = MyDiagnosis()
        #     mydia.diagnosis = request.user
        #     mydia.image = request.FILES['images']
        #     mydia.save()

        context = {
            'user': Users,
            'data': mydia
        }
        return render(request, 'que1.html', context)
        
    context = {
        'user': Users,
    }

    return render(request, 'que1.html', context)


def draw(request, user_pk):
    Users = User.objects.get(pk=user_pk)
    if request.POST:

        num = request.POST.getlist('imeg[]')
        if len(num) != 2:
            messages.info(request, '그림 두개를 골라주세요!')
            return redirect('/select/' + str(user_pk))
        else:
            num1 = num[0]
            num2 = num[1]
            # print(num1, num2)

            context = {
                'img1': num1+'.PNG',
                'img2': num2+'.PNG',
                'user': Users,
            }
            return render(request, 'draw.html', context)

    else:
        return redirect('/select/' + str(user_pk))
        
    return render(request, 'draw.html')

def select(request, user_pk):
    Users = User.objects.get(pk=user_pk)

    context = {
        'user': Users
    }

    return render(request, 'select.html', context)

def diagnosis(request, user_pk):
    Users = User.objects.get(pk=user_pk)

    context = {
        'user': Users
    }

    return render(request, 'diagnosis.html', context)

def mypage(request, user_pk):
    Users = User.objects.get(pk=user_pk)
    myaccount = Account.objects.get(user=Users)
    mydia = MyDiagnosis.objects.filter(diagnosis=Users)
    myrecord = Record.objects.filter(record_user=Users)
    record_elly = Record.objects.filter(writer='elly')

    # 진단날짜 클릭시 해당 db보여주기
    detail_dia = MyDiagnosis.objects.filter(diagnosis=Users)[:1].get()

    # 창원 작성_2020/12/04 16:31 
    # --------------------------------------------------------------
    MyDiagnosis_obj = MyDiagnosis.objects.all().values()
    df = pd.DataFrame(MyDiagnosis_obj)
    user_obj = df[df['diagnosis_id']==user_pk]
    user_obj.reset_index(inplace=True)
    # 데이터 프레임 만들기
        
    # 그래프로 그릴 항목 리스트 만들기
    created_date, score1, score2, score3, total_score = [], [], [], [], []
    # 각 항목 리스트 만들기
    for i in range(user_obj.shape[0]):
        created_date.append(str(user_obj['created_date'][i]))
        score1.append(float(user_obj['score1'][i]))
        score2.append(float(user_obj['score2'][i]))
        score3.append(float(user_obj['score3'][i]))
        total_score.append(float(user_obj['total_score'][i]))
    
    # 그래프 그리기
    plt.plot(created_date, score1, marker='o', label='정서 내용')
    plt.plot(created_date, score2, marker='x', label='자아상')
    plt.plot(created_date, score3, marker='^', label='자기보고')
    plt.plot(created_date, total_score, marker='+', label='최종점수')
    plt.legend(loc='upper right')
    plt.xlabel('날짜')
    plt.ylabel('점수')
    plt.title(str(Users)+'님의 진단 결과')
    plt.ylim([0,6])

    # 그래프 그림파일로 저장
    path = os.path.join(BASE_DIR, 'media/images/21_29')
    plt.savefig(path)
    paths = 'media/images/21_29' +'.png'
    plt.close()
    context = {
        'user': Users,
        'myaccount': myaccount,
        'mydia': mydia,
        'myrecord': myrecord,
        'record_elly' : record_elly,
        'detail_dia': detail_dia,
        'path': paths
    }
    return render(request, 'mypage.html', context) 

ERROR_MSG = {
    'ID_EXIST': '이미 사용 중인 아이디 입니다.',
    'ID_NOT_EXIST': '존재하지 않는 아이디 입니다',
    'ID_PW_MISSING': '아이디와 비밀번호를 다시 확인해주세요.',
    'PW_CHECK': '비밀번호가 일치하지 않습니다.',
}


def logout(request):
    auth.logout(request)
    return redirect('home')

def signup(request):
    context = {
        'error': {
            'state': False,
            'msg': ''
        }
    }
    if request.method == 'POST':
        
        user_id = request.POST['user_id']
        user_pw = request.POST['user_pw']
        user_pw_check = request.POST['user_pw_check']
        # add
        user_name = request.POST['name']
        user_nickname = request.POST['nickname']
        user_email = request.POST['email']
        user_gender = request.POST['gender']
        user_age = request.POST['age']

        if (user_id and user_pw):
            user = User.objects.filter(username=user_id)
            if len(user) == 0:
                if (user_pw == user_pw_check):
                    created_user = User.objects.create_user(
                        username=user_id,
                        password=user_pw
                    )
                    # add
                    Account.objects.create(
                        user = created_user,
                        name = user_name,
                        nickname = user_nickname,
                        email = user_email,
                        gender = user_gender,
                        age = user_age,
                    )

                    auth.login(request, created_user)
                    return redirect('home')
                else:
                    context['error']['state'] = True
                    context['error']['msg'] = ERROR_MSG['PW_CHECK']
            else:
                context['error']['state'] = True
                context['error']['msg'] = ERROR_MSG['ID_EXIST']
        else:
            context['error']['state'] = True
            context['error']['msg'] = ERROR_MSG['ID_PW_MISSING']

    return render(request, 'signup.html', context)

def login(request):
    context = {
        'error': {
            'state': False,
            'msg': ''
        },
    }
    if request.method == 'POST':
        user_id = request.POST['user_id']
        user_pw = request.POST['user_pw']

        user = User.objects.filter(username=user_id)
        if (user_id and user_pw):
            if len(user) != 0:
                user = auth.authenticate(
                    username=user_id,
                    password=user_pw
                )
                if user != None:
                    auth.login(request, user)

                    return redirect('home')
                else:
                    context['error']['state'] = True
                    context['error']['msg'] = ERROR_MSG['PW_CHECK']
            else:
                context['error']['state'] = True
                context['error']['msg'] = ERROR_MSG['ID_NOT_EXIST']
        else:
            context['error']['state'] = True
            context['error']['msg'] = ERROR_MSG['ID_PW_MISSING']

    return render(request, 'login.html', context)