"""ai URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
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
from django.urls import path,re_path,include
from app import views
from django.contrib.auth import views as auth_views
from django.views.generic import TemplateView
from app.views import *

urlpatterns = [
    path('admin/', admin.site.urls,name='admin'),
    re_path('^$', TemplateView.as_view(template_name='index.html'),name='index'),
    path('data/', DataView.as_view(), name='data'),
    path('create-data/', CreateDataView.as_view(), name='create_data'),
    path('add-database/', AddDatabaseView.as_view(), name='add_database'),
    path('show-table/', ShowTableView.as_view(), name='show_table'),
    path('sql-process/', SqlProcessView.as_view(), name='sql_process'),
    path('text-process/', TextProcessView.as_view(), name='text_process'),
    path('text-process-two/', TextProcessTwoView.as_view(), name='text_process_two'),
    path('con-data/', ConDataView.as_view(), name='con_data'),
    path('test/', TestView.as_view(), name='test'),
    # path('run/', RunView.as_view(), name='run'),
    path('sql-exe/', SqlExeView.as_view(), name='sql_exe'),
    path('kmeans/', views.kmeans, name='kemans'),
    re_path('table-edit/(?P<edit_id>\d+)$', views.table_edit, name='table_edit'),
    re_path('delete/(?P<delete_id>\d+)$', views.table_delete, name='table_delete'),
    path('project/', ProjectView.as_view(), name='project'),
    path('model/', Model2View.as_view(), name='model'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='logout.html'), name='logout'),
    path('register/', RegisterView.as_view(), name='register'),
    path('password-change/', auth_views.PasswordChangeView.as_view(template_name="password_change_form.html", success_url="/password-change-done/"), name='password_change'),
    path('password-change-done/', auth_views.PasswordChangeDoneView.as_view(template_name="password_change_done.html"), name='password_change_done'),
]
