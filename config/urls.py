"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from emotion import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home),
    # 회원가입
    path('join', views.join),
    # 로그인
    path('login', views.login),
    # 로그아웃
    path('logout', views.logout),
    path('emotion_test', views.emotion_test),
    # # 질문
    path('query', views.query),
    # # 채팅기록 삭제
    path('delete_chat', views.delete_chat),
]
