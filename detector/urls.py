from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect/', views.detect_image, name='detect_image'),
    path('clear/', views.clear_history, name='clear_history'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('result/<int:log_id>/', views.result_view, name='result_view'),
    path('delete_log/<int:log_id>/', views.delete_log, name='delete_log'),
    path('heatmap/<int:log_id>/', views.heatmap_detail_view, name='heatmap_detail'),
    path('history/', views.history_view, name='history'),
    path('api/history/', views.history_api, name='history_api'),
    path('export-csv/', views.export_csv, name='export_csv'),  # YEH NAYA ADD HUA HAI
    path('settings/', views.settings_view, name='settings'),
]