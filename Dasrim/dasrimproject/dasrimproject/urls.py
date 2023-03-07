"""dasrimproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.conf.urls.static import static
from django.conf import settings
import mainapp.views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', mainapp.views.home, name='home'),
    path('signup', mainapp.views.signup, name='signup'),
    path('login', mainapp.views.login, name='login'),
    path('logout', mainapp.views.logout, name='logout'),
    path('mypage/<int:user_pk>', mainapp.views.mypage, name='mypage'),
    path('diagnosis/<int:user_pk>', mainapp.views.diagnosis, name='diagnosis'),
    path('select/<int:user_pk>', mainapp.views.select, name='select'),
    path('draw/<int:user_pk>', mainapp.views.draw, name='draw'),
    path('que1/<int:user_pk>', mainapp.views.que1, name='que1'),
    path('que2/<int:user_pk>/<int:dia_pk>', mainapp.views.que2, name='que2'),
    path('que3/<int:user_pk>/<int:dia_pk>', mainapp.views.que3, name='que3'),
    path('result/<int:user_pk>/<int:dia_pk>', mainapp.views.result, name='result'),
    path('explain/<int:user_pk>', mainapp.views.explain, name='explain'),
    path('consult/<int:user_pk>', mainapp.views.consult, name='consult'),
    path('service/<int:user_pk>', mainapp.views.service, name='service'),
    path('record/<int:user_pk>', mainapp.views.record, name='record'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
