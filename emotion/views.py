from django.shortcuts import render, redirect
from emotion.models import Member, Predict
import hashlib
from emotion.myemotionbot import getMessage

# Create your views here.

def home(request):
    if 'userid' not in request.session.keys():
        return render(request, 'emotion/login.html')
    else:
        return render(request, 'emotion/main.html')

def login(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        passwd = hashlib.sha256(passwd.encode()).hexdigest()
        row = Member.objects.filter(userid=userid, passwd=passwd)
        # if row is not None:
        if row:
            row = row[0]
            request.session['userid'] = userid
            request.session['name'] = row.name
            return render(request, 'emotion/main.html')
        else:
            return render(request, 'emotion/login.html',
                          {'msg': '아이디 또는 비밀번호가 일치하지 않습니다.'})

    else:
        return render(request, 'emotion/login.html')

def join(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        passwd = hashlib.sha256(passwd.encode()).hexdigest()
        name = request.POST['name']
        address = request.POST['address']
        tel = request.POST['tel']
        Member(userid=userid, passwd=passwd, name=name, address=address, tel=tel).save()
        request.session['userid'] = userid
        request.session['name'] = name
        return render(request, 'emotion/main.html')
    else:
        return render(request, 'emotion/join.html')

def logout(request):
    request.session.clear()
    return redirect('/')

def emotion_test(request):
    return render(request, 'emotion/emotion_test.html')

def query(request):
    question = request.GET["question"]
    msg = getMessage(question)
    query = msg['Query']
    # print('query: ', query)
    predict = msg['predict']

    Predict(userid=request.session['userid'], query=query, predict=predict).save()
    # Predict(userid=request.session['userid'], predict=predict).save()
    # print('ddd')

    items=Predict.objects.filter(userid=request.session['userid']).order_by('-idx')
    # print('items', items)
    return render(request, 'emotion/result.html', {'items':items})


def delete_chat(request):
    Predict.objects.filter(userid=request.session['userid']).delete()
    return redirect('/emotion_test')

####
